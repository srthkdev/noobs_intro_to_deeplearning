"""Official M4 subset, four OFAT runs, and M5 annotation calculations.

All M4 training uses the existing M2 RGB SmallUNet without architectural changes.
Run with --analysis-only to reproduce data and toy calculations without training.
"""

from pathlib import Path
import argparse, json, hashlib, math, random, time, gc
from collections import Counter
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import train_test_split

CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
H, W = 128, 800


def decode_rle(value):
    flat = np.zeros(H * W, np.uint8)
    if str(value).strip():
        pairs = np.asarray(str(value).split(), dtype=np.int64).reshape(-1, 2)
        for start, length in pairs:
            assert start >= 1 and length > 0
            flat[start - 1 : min(start - 1 + length, H * W)] = 1
    return flat.reshape(H, W, order="F")


def schedule(t):
    return t / 60 if t < 60 else 0.5 * (1 + math.cos(math.pi * (t - 60) / 420))


def analyze(root, out):
    raw = pd.read_csv(root / "train.csv", keep_default_na=False)
    raw["mask_rle"] = raw.mask_rle.str.strip()
    full = (
        raw.pivot(index="image_id", columns="defect_type", values="mask_rle")
        .reindex(columns=CLASSES)
        .sort_index()
    )
    table = full.iloc[:1200].copy()
    assert len(table) == 1200 and len(full) > 1200
    presence = table.ne("")
    combos = presence.apply(
        lambda r: "+".join(c for c in CLASSES if r[c]) or "clean", axis=1
    )
    counts = combos.value_counts()
    strata = combos.where(combos.map(counts) >= 2, "rare_other")
    # rare_other still contains one image in the provided subset.
    # Keep that image in training and stratify all remaining images.
    singleton_mask = strata.map(strata.value_counts()) < 2
    forced_train = table.index[singleton_mask].to_numpy()
    eligible = table.index[~singleton_mask].to_numpy()
    train_regular, val_ids = train_test_split(
        eligible, test_size=240, random_state=42, stratify=strata.loc[eligible]
    )
    train_ids = np.concatenate([train_regular, forced_train])
    (out / "split-adjustment.json").write_text(
        json.dumps(
            {
                "reason": "rare_other still has one image, making exact stratification impossible",
                "forced_training_images": list(forced_train),
                "remaining_split": "stratified, random_state42, validation_size240",
                "train_size": len(train_ids),
                "validation_size": len(val_ids),
            },
            indent=2,
        )
    )
    assert (
        len(train_ids) == 960
        and len(val_ids) == 240
        and not set(train_ids) & set(val_ids)
    )
    masks = np.stack(
        [np.stack([decode_rle(x) for x in row]) for row in table.to_numpy()]
    )
    areas = masks.sum(axis=(-2, -1), dtype=np.int64)
    positive_areas = areas[areas > 0]
    b_pixels = int(areas[:, 1].sum())
    canvas_total = 1200 * H * W
    group_weights = {
        g: min(math.sqrt(int(counts["clean"]) / int(n)), 5.0) for g, n in counts.items()
    }
    weighted_counts = {g: int(counts[g]) * v for g, v in group_weights.items()}
    total_weight = sum(weighted_counts.values())
    natural_b = int(presence.defect_b.sum())
    sampler_b = (
        1200
        * sum(v for g, v in weighted_counts.items() if "defect_b" in g)
        / total_weight
    )
    small, medium, large = [
        int(x.sum())
        for x in [
            (positive_areas <= 100),
            ((positive_areas >= 101) & (positive_areas <= 500)),
            (positive_areas >= 501),
        ]
    ]
    missed = [0.7 * small, 0.3 * medium, 0.05 * large]
    stop_scores = [
        0.512,
        0.601,
        0.648,
        0.661,
        0.664,
        0.668,
        0.671,
        0.670,
        0.672,
        0.669,
        0.673,
        0.665,
    ]

    def early_trace(delta):
        best = -float("inf")
        counter = 0
        best_epoch = 0
        trace = []
        stop = None
        for epoch, score in enumerate(stop_scores, 1):
            improve = score > best + delta
            if improve:
                best = score
                best_epoch = epoch
                counter = 0
            else:
                counter += 1
            trace.append(
                dict(
                    epoch=epoch,
                    score=score,
                    improve=improve,
                    best_epoch=best_epoch,
                    best_score=best,
                    patience=counter,
                )
            )
            if counter >= 3:
                stop = epoch
                break
        return dict(stop=stop, best_epoch=best_epoch, best_score=best, trace=trace)

    answers = {
        "Q1": f"{counts['clean']}; "
        + ", ".join(str(int(presence[c].sum())) for c in CLASSES)
        + f"; {len(counts)}; {counts.max()}, {counts.min()}",
        "Q2": f"{100*b_pixels/canvas_total:.4f}%, {b_pixels/natural_b:.1f}, {round((canvas_total-b_pixels)/b_pixels)}",
        "Q3": f"{100*weighted_counts['clean']/total_weight:.2f}%, {100*counts['clean']/1200:.2f}%; {sampler_b:.2f}, {natural_b:.2f}; {sampler_b/natural_b:.2f}x; {sum(w==5 for w in group_weights.values())}",
        "Q4": ", ".join(f"{3e-4*schedule(t):.2e}" for t in [15, 270, 400])
        + f"; {3e-4*schedule(8):.2e}; The learning rate stays in the early warmup range, so learning is slow and the loss curves may look flat or underfit.",
        "Q5": f"{early_trace(.002)['stop']}, {early_trace(.002)['best_epoch']}, {early_trace(.002)['best_score']:.3f}; "
        + ("No" if early_trace(0)["stop"] is None else "Yes")
        + f", {early_trace(0)['best_epoch']}",
        "Q6": "400; 2800, 1600; 20, 40; 32; 120, 30; 4; 60",
        "Q7": f"{4800-int(presence.to_numpy().sum())}; {(4800-int(presence.to_numpy().sum()))/4800:.4f}; 0.3000; {(.7*.5+.2*.999+.1*.9):.4f}, {(.5*60+306)/400:.4f}; naive Dice, 0.8500",
        "Q9": f"{small}, {medium}, {large}; {100*positive_areas[positive_areas>=501].sum()/positive_areas.sum():.2f}%; {sum(missed):.2f}, {['1-100 px','101-500 px','501+ px'][int(np.argmax(missed))]}; Missing a few large defects can lose more pixels, while missing many tiny defects can dominate the number of missed pairs.",
    }
    stats = dict(
        subset_images=1200,
        train_images=960,
        validation_images=240,
        first_id=str(table.index[0]),
        last_id=str(table.index[-1]),
        subset_sha256=hashlib.sha256("\n".join(table.index).encode()).hexdigest(),
        positive_counts={c: int(presence[c].sum()) for c in CLASSES},
        combination_counts={str(k): int(v) for k, v in counts.items()},
        defect_b_pixels=b_pixels,
        defect_b_negative_positive_ratio=(canvas_total - b_pixels) / b_pixels,
        all_empty_naive=(4800 - int(presence.to_numpy().sum())) / 4800,
        positive_bins=[small, medium, large],
        positive_bin_pixels=[
            int(positive_areas[(positive_areas >= a) & (positive_areas <= b)].sum())
            for a, b in [(1, 100), (101, 500), (501, H * W)]
        ],
        early_stop_delta002=early_trace(0.002),
        early_stop_delta0=early_trace(0),
    )
    (out / "analysis.json").write_text(json.dumps(stats, indent=2))
    (out / "answers-static.json").write_text(json.dumps(answers, indent=2))
    pd.DataFrame(
        {
            "image_id": table.index,
            "combination": combos,
            "stratum": strata,
            "split": [
                "train" if x in set(train_ids) else "validation" for x in table.index
            ],
        }
    ).to_csv(out / "subset-split.csv", index=False)
    pd.DataFrame(
        [
            dict(
                group=g, count=int(n), weight=group_weights[g], mass=weighted_counts[g]
            )
            for g, n in counts.items()
        ]
    ).to_csv(out / "sampler-analysis.csv", index=False)
    # M5 Q5 explicitly requests every positive pair in the full official training CSV.
    wiped = Counter()
    total_positive = 0
    for _, row in full.iterrows():
        for c in CLASSES:
            if not row[c]:
                continue
            total_positive += 1
            lab, n = ndimage.label(
                decode_rle(row[c]), structure=np.ones((3, 3), np.uint8)
            )
            sizes = np.bincount(lab.ravel())[1:]
            if len(sizes) == 0 or sizes.max() < 100:
                wiped[c] += 1
    (out / "milestone5-recall-ceiling.json").write_text(
        json.dumps(
            dict(
                total_positive_pairs=total_positive,
                wiped_pairs=sum(wiped.values()),
                wiped_by_class={c: wiped[c] for c in CLASSES},
                worst_class=max(CLASSES, key=lambda c: wiped[c]),
                connectivity=8,
                minimum_area=100,
            ),
            indent=2,
        )
    )
    print(json.dumps(answers, indent=2), flush=True)
    return table, combos, train_ids, val_ids, masks


def train(root, out, table, combos, train_ids, val_ids, masks):
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from PIL import Image
    import wandb

    assert torch.cuda.is_available(), "Official fixed setup requires CUDA AMP"
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    images = np.stack(
        [
            np.asarray(
                Image.open(root / "train" / (str(i) + ".jpg")).convert("RGB"), np.uint8
            )
            for i in table.index
        ]
    )
    locations = {x: i for i, x in enumerate(table.index)}

    class Strips(Dataset):
        def __init__(self, ids, augment):
            self.idx = [locations[i] for i in ids]
            self.augment = augment

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            x = images[j].astype(np.float32) / 255.0
            y = masks[j].astype(np.float32)
            if self.augment:
                if random.random() < 0.5:
                    x = x[:, ::-1]
                    y = y[:, :, ::-1]
                if random.random() < 0.5:
                    x = x[::-1]
                    y = y[:, ::-1, :]
                if random.random() < 0.5:
                    x = np.clip(
                        x * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05), 0, 1
                    )
            return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(
                y.copy()
            )

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.layers(x)

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = ConvBlock(3, 16)
            self.enc2 = ConvBlock(16, 32)
            self.enc3 = ConvBlock(32, 64)
            self.bottleneck = ConvBlock(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.dec3 = ConvBlock(128 + 64, 64)
            self.dec2 = ConvBlock(64 + 32, 32)
            self.dec1 = ConvBlock(32 + 16, 16)
            self.head = nn.Conv2d(16, 4, 1)

        def forward(self, x):
            a = self.enc1(x)
            b = self.enc2(self.pool(a))
            c = self.enc3(self.pool(b))
            x = self.bottleneck(self.pool(c))
            for skip, decoder in [(c, self.dec3), (b, self.dec2), (a, self.dec1)]:
                x = nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
                x = decoder(torch.cat([x, skip], dim=1))
            return self.head(x)

    def criterion(logits, y):
        logits = logits.float()
        p = logits.sigmoid()
        intersection = (p * y).sum((-2, -1))
        den = (p + y).sum((-2, -1))
        positive = y.sum((-2, -1)) > 0
        dice = 1 - (2 * intersection + 1e-6) / (den + 1e-6)
        return 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, y) + 0.5 * (
            dice * positive
        ).sum() / positive.sum().clamp_min(1)

    def evaluate(model, loader):
        model.eval()
        loss_sum = 0.0
        dice_sum = 0.0
        fg_sum = 0.0
        fp_sum = 0.0
        clean_correct = 0
        n = pos = empty = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    z = model(x)
                    loss = criterion(z, y)
                truth = y > 0.5
                pred = z.float().sigmoid() > 0.5
                area = truth.sum((-2, -1))
                pa = pred.sum((-2, -1))
                inter = (truth & pred).sum((-2, -1))
                den = area + pa
                d = torch.where(
                    den == 0,
                    torch.ones_like(den, dtype=torch.float32),
                    2 * inter / den.clamp_min(1),
                )
                positive = area > 0
                e = ~positive
                loss_sum += loss.item() * len(x)
                n += len(x)
                pos += int(positive.sum())
                empty += int(e.sum())
                fg_sum += float(d[positive].sum())
                dice_sum += float(d.sum())
                fp_sum += float((pa[e].float() / (H * W)).sum())
                clean_correct += int((e & (pa == 0)).sum())
        fg = fg_sum / pos
        fp = 1 - fp_sum / empty
        ea = clean_correct / empty
        return dict(
            val_loss=loss_sum / n,
            official_score=0.7 * fg + 0.2 * fp + 0.1 * ea,
            foreground_dice=fg,
            false_positive_score=fp,
            empty_accuracy=ea,
            empty_false_alarm_rate=1 - ea,
            naive_dice=dice_sum / (4 * n),
            positive_pairs=pos,
            empty_pairs=empty,
        )

    runs = []
    for name, seed, warm, sampling in [
        ("R0", 42, False, False),
        ("R0b", 7, False, False),
        ("R1", 42, True, False),
        ("R2", 42, False, True),
    ]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SmallUNet().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule) if warm else None
        generator = torch.Generator().manual_seed(seed)
        sampler = None
        if sampling:
            train_groups = combos.loc[train_ids]
            counts = train_groups.value_counts()
            weights = [
                min(math.sqrt(int(counts["clean"]) / int(counts[g])), 5.0)
                for g in train_groups
            ]
            sampler = WeightedRandomSampler(
                weights, num_samples=960, replacement=True, generator=generator
            )
            pd.DataFrame(
                {
                    "image_id": train_ids,
                    "combination": train_groups.to_numpy(),
                    "sampler_weight": weights,
                }
            ).to_csv(out / "R2-train-sampler.csv", index=False)
        loader = DataLoader(
            Strips(train_ids, True),
            batch_size=16,
            shuffle=sampler is None,
            sampler=sampler,
            generator=torch.Generator().manual_seed(seed),
            num_workers=2,
            pin_memory=True,
        )
        val = DataLoader(
            Strips(val_ids, False),
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        assert len(loader) == 60
        cfg = dict(
            run=name,
            seed=seed,
            epochs=6,
            batch_size=16,
            optimizer="AdamW",
            lr=3e-4,
            weight_decay=1e-4,
            AMP=True,
            input_shape=[3, H, W],
            baseline="M2 RGB SmallUNet",
            normalization="divide by 255",
            augmentation="same as M2: joint flips and brightness/contrast",
            loss="0.5 BCE + 0.5 positive-pair soft Dice",
            schedule="60-step warmup + cosine denominator420" if warm else "constant",
            schedule_reference_steps=480 if warm else None,
            sampling=(
                "bounded square-root on training combinations, cap5"
                if sampling
                else "natural uniform shuffled"
            ),
            decision_threshold=0.5,
            strict_comparison=">",
            min_area=0,
            train_images=960,
            val_images=240,
            split_seed=42,
            questions=["Q8", "Q10"] if name == "R1" else ["Q10"],
        )
        run = wandb.init(
            entity="23f3000839-iit-madras-foundation",
            project="23f3000839-t32026",
            name=f"milestone4-{name}-subset1200",
            mode="offline",
            dir=str(out),
            config=cfg,
            reinit=True,
        )
        scaler = torch.amp.GradScaler("cuda")
        history = []
        best = -float("inf")
        best_epoch = None
        best_row = None
        run_dir = out / name
        run_dir.mkdir(exist_ok=True)
        for epoch in range(1, 7):
            model.train()
            total = 0.0
            count = 0
            start = time.perf_counter()
            used_lrs = []
            for x, y in loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                used_lrs.append(float(opt.param_groups[0]["lr"]))
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if scheduler:
                    scheduler.step()
                total += loss.item() * len(x)
                count += len(x)
            row = dict(
                epoch=epoch,
                train_loss=total / count,
                **evaluate(model, val),
                lr_first=used_lrs[0],
                lr_last=used_lrs[-1],
                seconds=time.perf_counter() - start,
            )
            history.append(row)
            run.log(row)
            print(name, json.dumps(row), flush=True)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            if row["official_score"] > best:
                best = row["official_score"]
                best_epoch = epoch
                best_row = row.copy()
                torch.save(model.state_dict(), run_dir / "best.pt")
        model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True))
        restored = evaluate(model, val)
        assert abs(restored["official_score"] - best) < 1e-8
        summary = dict(
            name=name,
            seed=seed,
            best_epoch=best_epoch,
            best_official_score=best,
            best_metrics=best_row,
            best_val_loss_epoch=min(history, key=lambda r: r["val_loss"])["epoch"],
            wandb_id=run.id,
            config=cfg,
            restored_metrics=restored,
        )
        run.summary.update(
            {
                "best_official_score": best,
                "best_epoch": best_epoch,
                "best_empty_false_alarm_rate": best_row["empty_false_alarm_rate"],
            }
        )
        run.finish()
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        runs.append(summary)
        del model, opt, loader, val, scaler
        gc.collect()
        torch.cuda.empty_cache()
    (out / "training-summary.json").write_text(json.dumps(runs, indent=2))
    rr = {r["name"]: r for r in runs}
    noise = abs(rr["R0"]["best_official_score"] - rr["R0b"]["best_official_score"])
    gains = {
        n: rr[n]["best_official_score"] - rr["R0"]["best_official_score"]
        for n in ["R1", "R2"]
    }
    h = pd.read_csv(out / "R1/history.csv")
    last = h.iloc[-1]
    q8 = f"{int(h.loc[h.val_loss.idxmin(),'epoch'])}; {rr['R1']['best_epoch']}, {rr['R1']['best_official_score']:.3f}; {last.val_loss-last.train_loss:.3f}; "
    q8 += f"I would select by the official validation score and keep epoch {rr['R1']['best_epoch']}, because that is the metric used for evaluation. "
    q8 += "The learning-curve pattern needs to be checked against foreground Dice as well as loss; the notebook includes that interpretation after the run."
    q10 = (
        ", ".join(
            f"{rr[n]['best_official_score']:.3f}" for n in ["R0", "R0b", "R1", "R2"]
        )
        + f"; {noise:.3f}, {gains['R1']:.3f}, {gains['R2']:.3f}; "
        + ", ".join("yes" if gains[n] > 2 * noise else "no" for n in ["R1", "R2"])
        + f"; {100*rr['R0']['best_metrics']['empty_false_alarm_rate']:.1f}%, {100*rr['R2']['best_metrics']['empty_false_alarm_rate']:.1f}%; If several settings change together, I cannot attribute the score change to one of them. "
    )
    q10 += "The sampler changes the defect frequency seen during training; its false-alarm change shows how that training prior affects predictions on naturally distributed clean pairs."
    a = json.loads((out / "answers-static.json").read_text())
    a.update(Q8=q8, Q10=q10)
    (out / "answers.json").write_text(json.dumps(a, indent=2))
    print("FINAL ANSWERS", json.dumps(a, indent=2), flush=True)
    pd.DataFrame(
        [
            dict(
                run=r["name"],
                best_epoch=r["best_epoch"],
                score=r["best_official_score"],
                false_alarm=r["best_metrics"]["empty_false_alarm_rate"],
            )
            for r in runs
        ]
    ).to_csv(out / "run-comparison.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for r in runs:
        h = pd.read_csv(out / r["name"] / "history.csv")
        axes[0].plot(h.epoch, h.official_score, marker="o", label=r["name"])
    h = pd.read_csv(out / "R1/history.csv")
    axes[1].plot(h.epoch, h.train_loss, label="train loss")
    axes[1].plot(h.epoch, h.val_loss, label="validation loss")
    axes[0].set_ylabel("Official validation score")
    axes[1].set_ylabel("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training-curves.png", dpi=160)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--analysis-only", action="store_true")
    args = p.parse_args()
    root = args.data or next(
        p.parent
        for p in Path("/kaggle/input").rglob("train.csv")
        if (p.parent / "train").is_dir()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    table, combos, train_ids, val_ids, masks = analyze(root, args.output)
    if not args.analysis_only:
        train(root, args.output, table, combos, train_ids, val_ids, masks)


if __name__ == "__main__":
    main()
