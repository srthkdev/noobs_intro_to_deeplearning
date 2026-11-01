"""Three-model surface segmentation experiment. No credentials are stored in code.

Kaggle: python final_pipeline.py --data /kaggle/input/.../public --output /kaggle/working
Local: use the same entry point and a CUDA/MPS/CPU device selected automatically.
"""

import argparse, copy, gc, json, os, random, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import wandb

CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
H, W = 128, 800
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_rle(value):
    flat = np.zeros(H * W, dtype=np.uint8)
    if str(value).strip():
        a = np.asarray(str(value).split(), dtype=np.int64)
        assert len(a) % 2 == 0
        for start, length in a.reshape(-1, 2):
            assert start >= 1 and length > 0 and start + length - 1 <= H * W
            flat[start - 1 : start - 1 + length] = 1
    return flat.reshape(H, W, order="F")


def encode_rle(mask):
    flat = mask.astype(np.uint8).ravel(order="F")
    changes = np.flatnonzero(np.diff(np.r_[0, flat, 0])) + 1
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes)) or " "


def remove_components(mask, min_area):
    if not min_area:
        return mask
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


class StripDataset(Dataset):
    def __init__(self, ids, table, root, training=False):
        self.ids = list(ids)
        self.table = table
        self.root = Path(root)
        self.training = training

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image_id = self.ids[i]
        filename = image_id if image_id.endswith(".jpg") else image_id + ".jpg"
        with Image.open(
            self.root / ("train" if self.table is not None else "test") / filename
        ) as f:
            x = np.asarray(f.convert("L"), dtype=np.float32) / 255
        y = (
            np.stack([decode_rle(self.table.loc[image_id, c]) for c in CLASSES])
            if self.table is not None
            else np.zeros((4, H, W), dtype=np.uint8)
        )
        assert x.shape == (H, W)
        if self.training:
            if random.random() < 0.5:
                x = x[:, ::-1]
                y = y[:, :, ::-1]
            if random.random() < 0.5:
                x = x[::-1, :]
                y = y[:, ::-1, :]
            if random.random() < 0.5:
                x = np.clip(
                    x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08), 0, 1
                )
        # Keep grayscale intensity information; match training and inference exactly.
        x = (x - 0.5) / 0.25
        return torch.from_numpy(x[None].copy()), torch.from_numpy(
            y.astype(np.float32).copy()
        )


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ScratchUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList(
            [Block(1, 24), Block(24, 48), Block(48, 96), Block(96, 192)]
        )
        self.dec = nn.ModuleList([Block(288, 96), Block(144, 48), Block(72, 24)])
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(24, 4, 1)

    def forward(self, x):
        skips = []
        for i, b in enumerate(self.enc):
            x = b(x)
            skips.append(x)
            if i < 3:
                x = self.pool(x)
        for skip, b in zip(skips[-2::-1], self.dec):
            x = b(
                torch.cat(
                    [
                        nn.functional.interpolate(
                            x,
                            size=skip.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        skip,
                    ],
                    dim=1,
                )
            )
        return self.head(x)


def make_model(name, pretrained=True):
    if name == "scratch_unet":
        return ScratchUNet()
    import segmentation_models_pytorch as smp

    if name == "resnet34_unet":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=1,
            classes=4,
        )
    if name == "efficientnet_b0_fpn":
        return smp.FPN(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=1,
            classes=4,
            decoder_pyramid_channels=128,
            decoder_segmentation_channels=64,
        )
    raise ValueError(name)


def loss_fn(logits, y):
    logits = logits.float()
    p = logits.sigmoid()
    dims = (-2, -1)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
    intersection = (p * y).sum(dims)
    denom = (p + y).sum(dims)
    positive = y.sum(dims) > 0
    dice = 1 - (2 * intersection + 1) / (denom + 1)
    # Equal average over positive pairs prevents the abundant clean masks dominating Dice.
    dice = (dice * positive).sum() / positive.sum().clamp_min(1)
    return 0.5 * bce + 0.5 * dice


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        total += loss.item() * len(x)
        n += len(x)
    return total / n


@torch.no_grad()
def cache_probabilities(model, loader, device, destination, tta=False):
    model.eval()
    count = len(loader.dataset)
    probs = np.lib.format.open_memmap(
        destination, mode="w+", dtype="float16", shape=(count, 4, H, W)
    )
    cursor = 0
    for x, _ in loader:
        x = x.to(device)
        p = model(x).sigmoid()
        if tta:
            p = (p + torch.flip(model(torch.flip(x, [-1])).sigmoid(), [-1])) / 2
        n = len(x)
        probs[cursor : cursor + n] = p.float().cpu().numpy().astype(np.float16)
        cursor += n
    probs.flush()
    return probs


def score_predictions(probs, truth, thresholds, areas):
    accum = np.zeros((4, 9), dtype=np.float64)
    for i in range(len(probs)):
        for c in range(4):
            pred = remove_components(np.asarray(probs[i, c]) >= thresholds[c], areas[c])
            gt = truth[i, c] > 0
            a, b = int(pred.sum()), int(gt.sum())
            tp = int((pred & gt).sum())
            den = a + b
            dice = 2 * tp / den if den else 1.0
            v = accum[c]
            v[0] += dice
            v[1] += b > 0
            v[2] += dice if b else 0
            v[3] += b == 0
            v[4] += a / (H * W) if not b else 0
            v[5] += a == 0 and b == 0
            v[6] += tp
            v[7] += a
            v[8] += b
    rows = []
    for c, v in enumerate(accum):
        fg = v[2] / max(v[1], 1)
        fp = v[4] / max(v[3], 1)
        empty = v[5] / max(v[3], 1)
        rows.append(
            {
                "class": CLASSES[c],
                "dice": v[0] / len(probs),
                "foreground_dice": fg,
                "empty_accuracy": empty,
                "fp_pixel_fraction": fp,
                "assessment_score": 0.7 * fg + 0.2 * (1 - fp) + 0.1 * empty,
                "pixel_f1": 2 * v[6] / max(v[7] + v[8], 1),
                "pixel_accuracy": (len(probs) * H * W - v[7] - v[8] + 2 * v[6])
                / (len(probs) * H * W),
            }
        )
    return pd.DataFrame(rows)


def build_truth(ids, table, path):
    out = np.lib.format.open_memmap(
        path, mode="w+", dtype="uint8", shape=(len(ids), 4, H, W)
    )
    for i, image_id in enumerate(ids):
        out[i] = np.stack([decode_rle(table.loc[image_id, c]) for c in CLASSES])
    out.flush()
    return out


def tune_postprocess(probs, truth):
    # Nine probability/area combinations per class; biologically unrelated giant searches avoided.
    rows = []
    thresholds = []
    areas = []
    for c in range(4):
        best = (-1, None, None)
        for tau in [0.35, 0.5, 0.65]:
            for area in [0, 32, 128]:
                # Evaluate this class only using a light view of the required channels.
                fg = fp = empty_ok = positive = empty = 0.0
                for i in range(len(probs)):
                    pred = remove_components(probs[i, c] >= tau, area)
                    gt = truth[i, c] > 0
                    a, b = int(pred.sum()), int(gt.sum())
                    if b:
                        fg += 2 * int((pred & gt).sum()) / max(a + b, 1)
                        positive += 1
                    else:
                        fp += a / (H * W)
                        empty_ok += a == 0
                        empty += 1
                s = (
                    0.7 * fg / max(positive, 1)
                    + 0.2 * (1 - fp / max(empty, 1))
                    + 0.1 * empty_ok / max(empty, 1)
                )
                rows.append(
                    {
                        "class": CLASSES[c],
                        "threshold": tau,
                        "min_area": area,
                        "tuning_assessment_score": s,
                    }
                )
                if s > best[0]:
                    best = (s, tau, area)
        thresholds.append(best[1])
        areas.append(best[2])
    return thresholds, areas, pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument(
        "--models",
        nargs="+",
        default=["scratch_unet", "resnet34_unet", "efficientnet_b0_fpn"],
    )
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if args.data is None:
        candidates = [
            q.parent
            for q in Path("/kaggle/input").rglob("train.csv")
            if (q.parent / "train").is_dir() and (q.parent / "test").is_dir()
        ]
        assert len(candidates) == 1, candidates
        args.data = candidates[0]
    seed_everything()
    torch.set_num_threads(4)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(
        "Device:", device, "Torch:", torch.__version__, "Data:", args.data, flush=True
    )
    args.output.mkdir(parents=True, exist_ok=True)
    table = (
        pd.read_csv(args.data / "train.csv", keep_default_na=False)
        .pivot(index="image_id", columns="defect_type", values="mask_rle")
        .reindex(columns=CLASSES)
        .sort_index()
        .apply(lambda col: col.str.strip())
    )
    combo = table.ne("").apply(lambda r: "".join(str(int(v)) for v in r), axis=1)
    train_ids, holdout_ids = train_test_split(
        table.index.to_numpy(), test_size=0.2, stratify=combo, random_state=SEED
    )
    # Fixed M1 outer holdout retained. Split it into postprocess tuning and reporting portions.
    tune_ids, audit_ids = train_test_split(
        holdout_ids,
        test_size=0.5,
        stratify=combo.loc[holdout_ids],
        random_state=SEED + 1,
    )
    split = pd.DataFrame({"image_id": table.index, "partition": "train"}).set_index(
        "image_id"
    )
    split.loc[tune_ids, "partition"] = "tune"
    split.loc[audit_ids, "partition"] = "audit"
    split.to_csv(args.output / "split.csv")
    assert not set(train_ids) & set(holdout_ids)
    training = StripDataset(train_ids, table, args.data, True)
    validation = StripDataset(holdout_ids, table, args.data)
    counts = table.loc[train_ids].ne("").sum().to_numpy()
    presence = table.loc[train_ids].ne("").to_numpy()
    # Temper rather than fully equalize inverse prevalence to limit duplicate exposure.
    weights = np.maximum(
        1.0, (presence * np.sqrt(len(train_ids) / np.maximum(counts, 1))).max(axis=1)
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        len(training),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 2 if device.type == "cuda" else 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(training, sampler=sampler, **kwargs)
    val_loader = DataLoader(validation, shuffle=False, **kwargs)
    tune_loader = DataLoader(
        StripDataset(tune_ids, table, args.data), shuffle=False, **kwargs
    )
    audit_loader = DataLoader(
        StripDataset(audit_ids, table, args.data), shuffle=False, **kwargs
    )
    tune_truth = build_truth(tune_ids, table, args.output / "tune_truth.npy")
    audit_truth = build_truth(audit_ids, table, args.output / "audit_truth.npy")
    comparison = []
    artifacts = []
    for name in args.models:
        seed_everything()
        run_dir = args.output / name
        run_dir.mkdir(exist_ok=True)
        model = make_model(name).to(device)
        lr = 0.001 if name == "scratch_unet" else 0.0003
        config = {
            "architecture": name,
            "pretrained": name != "scratch_unet",
            "seed": SEED,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "height": H,
            "width": W,
            "loss": "0.5 BCE + 0.5 positive pair Dice",
            "augmentation": "horizontal/vertical flips; brightness/contrast",
            "sampler": "tempered inverse prevalence",
            "train_images": len(train_ids),
            "tune_images": len(tune_ids),
            "audit_images": len(audit_ids),
            "lr": lr,
            "device": str(device),
        }
        # Offline logs are synced with local account authentication after download; no key in notebook.
        run = wandb.init(
            entity="23f3000839-iit-madras-foundation",
            project="23f3000839-t32026",
            name=f"{name}-final-seed42",
            job_type="training",
            mode="offline",
            dir=str(args.output),
            config=config,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=lr / 30
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best = float("inf")
        bad = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            model.train()
            total = 0.0
            n = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * len(x)
                n += len(x)
            # Early stopping uses tune images only; audit data never selects checkpoints.
            val = evaluate_loss(model, tune_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": total / n,
                "tune_loss": val,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - start,
            }
            history.append(row)
            run.log(row)
            print(name, row, flush=True)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            scheduler.step()
            if val < best - 1e-4:
                best = val
                bad = 0
                torch.save(
                    {
                        "architecture": name,
                        "state_dict": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                    },
                    run_dir / "best.pt",
                )
            else:
                bad += 1
            if bad >= 7:
                break
        checkpoint = torch.load(
            run_dir / "best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        probs = cache_probabilities(
            model, tune_loader, device, run_dir / "tune_probs.npy"
        )
        tau, area, trials = tune_postprocess(probs, tune_truth)
        trials.to_csv(run_dir / "postprocessing_trials.csv", index=False)
        tune_scores = score_predictions(probs, tune_truth, tau, area)
        tune_scores.to_csv(run_dir / "tune_metrics.csv", index=False)
        audit = cache_probabilities(
            model, audit_loader, device, run_dir / "audit_probs.npy"
        )
        audit_scores = score_predictions(audit, audit_truth, tau, area)
        audit_scores.to_csv(run_dir / "audit_metrics.csv", index=False)
        row = {
            "model": name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": checkpoint["epoch"],
            "tune_loss": best,
            "tune_assessment_score": float(tune_scores.assessment_score.mean()),
            "audit_assessment_score": float(audit_scores.assessment_score.mean()),
            "audit_macro_pixel_f1": float(audit_scores.pixel_f1.mean()),
            "audit_pixel_accuracy": float(audit_scores.pixel_accuracy.mean()),
            "thresholds": tau,
            "min_areas": area,
            "wandb_run_id": run.id,
        }
        comparison.append(row)
        (run_dir / "summary.json").write_text(json.dumps(row, indent=2))
        pd.DataFrame(comparison).to_csv(
            args.output / "model_comparison.csv", index=False
        )
        run.summary.update(row)
        run.log({"audit_metrics": wandb.Table(dataframe=audit_scores)})
        run.finish()
        artifacts.append((name, tau, area))
        del model, optimizer, scaler, probs, audit
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Select by tuning score, then report audit once for the selected model. No leaderboard-based search.
    selected = max(comparison, key=lambda r: r["tune_assessment_score"])
    (args.output / "selection.json").write_text(json.dumps(selected, indent=2))
    model = make_model(selected["model"], pretrained=False).to(device)
    model.load_state_dict(
        torch.load(
            args.output / selected["model"] / "best.pt",
            map_location=device,
            weights_only=False,
        )["state_dict"]
    )
    sample = pd.read_csv(args.data / "sample_submission.csv", keep_default_na=False)
    test_ids = sample.image_id.unique()
    test_loader = DataLoader(
        StripDataset(test_ids, None, args.data), shuffle=False, **kwargs
    )
    probs = cache_probabilities(
        model, test_loader, device, args.output / "test_probs.npy", tta=False
    )
    predictions = {}
    for i, image_id in enumerate(test_ids):
        for c, label in enumerate(CLASSES):
            predictions[(image_id, label)] = encode_rle(
                remove_components(
                    probs[i, c] >= selected["thresholds"][c], selected["min_areas"][c]
                )
            )
    sample["mask_rle"] = [
        predictions[(r.image_id, r.defect_type)] for r in sample.itertuples()
    ]
    assert not sample.isna().any().any()
    sample.to_csv(args.output / "submission.csv", index=False)
    # Keep checkpoints, logs and report metrics; discard large probability arrays after use.
    for path in args.output.rglob("*.npy"):
        path.unlink()
    print("COMPLETED", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
