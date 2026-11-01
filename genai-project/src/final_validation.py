"""Compare three prespecified inference variants without leaderboard tuning.

Variants: selected single model, that model with horizontal-flip TTA, and the
unweighted probability mean of all three trained architectures. Choose by the
same tuning metric, preferring fewer forward passes within 0.002 of the best.
"""

import argparse
import gc
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
import wandb
from final_pipeline import (
    CLASSES,
    H,
    W,
    StripDataset,
    make_model,
    cache_probabilities,
    build_truth,
    tune_postprocess,
    score_predictions,
    remove_components,
    encode_rle,
)


def load_model(name, model_root, device):
    saved = torch.load(
        model_root / name / "best.pt", map_location=device, weights_only=False
    )
    assert saved["architecture"] == name
    model = make_model(name, pretrained=False).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def average_files(paths, destination):
    files = [np.load(p, mmap_mode="r") for p in paths]
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype="float16", shape=files[0].shape
    )
    for start in range(0, len(output), 16):
        end = start + 16
        output[start:end] = sum(
            np.asarray(p[start:end], dtype=np.float32) for p in files
        ) / len(files)
    output.flush()
    return output


def pair_metrics(probs, truth, ids, thresholds, areas):
    rows = []
    for i, image_id in enumerate(ids):
        for c, label in enumerate(CLASSES):
            prediction = remove_components(probs[i, c] >= thresholds[c], areas[c])
            target = truth[i, c] > 0
            p, g = int(prediction.sum()), int(target.sum())
            intersection = int((prediction & target).sum())
            rows.append(
                dict(
                    image_id=image_id,
                    defect_type=label,
                    prediction_pixels=p,
                    target_pixels=g,
                    dice=2 * intersection / (p + g) if p + g else 1.0,
                )
            )
    return pd.DataFrame(rows)


def error_figure(pairs, probs, truth, ids, thresholds, areas, data_root, destination):
    chosen = []
    for label in CLASSES:
        eligible = pairs[(pairs.defect_type == label) & (pairs.target_pixels > 0)]
        if len(eligible):
            chosen.append(eligible.sort_values("dice").iloc[0])
    empty = pairs[(pairs.target_pixels == 0) & (pairs.prediction_pixels > 0)]
    if len(empty):
        chosen.append(empty.sort_values("prediction_pixels", ascending=False).iloc[0])
    lookup = {image_id: i for i, image_id in enumerate(ids)}
    fig, axes = plt.subplots(
        len(chosen), 2, figsize=(8.5, 1.2 * len(chosen)), squeeze=False
    )
    selected_rows = []
    for row_idx, row in enumerate(chosen):
        i, c = lookup[row.image_id], CLASSES.index(row.defect_type)
        with Image.open(data_root / "train" / (row.image_id + ".jpg")) as image:
            gray = np.asarray(image.convert("L"))
        prediction = remove_components(probs[i, c] >= thresholds[c], areas[c])
        for col, (mask, title) in enumerate(
            [(truth[i, c] > 0, "Ground truth"), (prediction, "Prediction")]
        ):
            rgb = np.repeat(gray[..., None], 3, axis=-1).astype(float)
            rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.array([235, 65, 55])
            axes[row_idx, col].imshow(rgb.astype("uint8"))
            axes[row_idx, col].set_title(
                f"{row.defect_type}: {title}; Dice={row.dice:.3f}", fontsize=10
            )
            axes[row_idx, col].axis("off")
        selected_rows.append(row.to_dict())
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return selected_rows


def predict_test(decision, model_root, data_root, output, device, batch_size):
    sample = pd.read_csv(data_root / "sample_submission.csv", keep_default_na=False)
    ids = sample.image_id.unique()
    loader = DataLoader(
        StripDataset(ids, None, data_root),
        batch_size=batch_size,
        num_workers=2 if device.type == "cuda" else 0,
        shuffle=False,
    )
    paths = []
    for name in decision["model_names"]:
        model = load_model(name, model_root, device)
        path = output / (name + "-test.npy")
        cache_probabilities(model, loader, device, path, tta=decision["tta"])
        paths.append(path)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    probs = (
        average_files(paths, output / "final-test.npy")
        if len(paths) > 1
        else np.load(paths[0], mmap_mode="r")
    )
    rles = {}
    for i, image_id in enumerate(ids):
        for c, label in enumerate(CLASSES):
            mask = remove_components(
                probs[i, c] >= decision["thresholds"][c], decision["min_areas"][c]
            )
            rles[(image_id, label)] = encode_rle(mask)
    sample["mask_rle"] = [
        rles[(r.image_id, r.defect_type)] for r in sample.itertuples()
    ]
    assert len(sample) == len(ids) * 4 and not sample.isna().any().any()
    sample.to_csv(output / "submission.csv", index=False)
    return len(sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.data is None:
        roots = [
            p.parent
            for p in Path("/kaggle/input").rglob("train.csv")
            if (p.parent / "train").is_dir() and (p.parent / "test").is_dir()
        ]
        assert len(roots) == 1, roots
        args.data = roots[0]
    if args.models is None:
        roots = [
            p.parent
            for p in Path("/kaggle/input").rglob("selection.json")
            if (p.parent / "model_comparison.csv").exists()
        ]
        assert len(roots) == 1, roots
        args.models = roots[0]
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.set_num_threads(4)
    data = pd.read_csv(args.data / "train.csv", keep_default_na=False)
    table = (
        data.pivot(index="image_id", columns="defect_type", values="mask_rle")
        .reindex(columns=CLASSES)
        .apply(lambda c: c.str.strip())
    )
    split = pd.read_csv(args.models / "split.csv").set_index("image_id")
    # Dataset order must match probability caches exactly, not sorted annotation order.
    tune_ids = split.index[split.partition.eq("tune")].tolist()
    audit_ids = split.index[split.partition.eq("audit")].tolist()
    truth = {
        part: build_truth(ids, table, args.output / f"{part}-truth.npy")
        for part, ids in [("tune", tune_ids), ("audit", audit_ids)]
    }
    loaders = {
        part: DataLoader(
            StripDataset(ids, table, args.data),
            batch_size=args.batch_size,
            num_workers=2 if device.type == "cuda" else 0,
            shuffle=False,
        )
        for part, ids in [("tune", tune_ids), ("audit", audit_ids)]
    }
    training = pd.read_csv(args.models / "model_comparison.csv")
    names = training.model.tolist()
    assert set(names) == {"scratch_unet", "resnet34_unet", "efficientnet_b0_fpn"}
    selected = json.loads((args.models / "selection.json").read_text())["model"]
    paths = {}
    timings = []
    for name in names:
        model = load_model(name, args.models, device)
        start = time.perf_counter()
        for part in ["tune", "audit"]:
            path = args.output / f"{name}-{part}.npy"
            cache_probabilities(model, loaders[part], device, path)
            paths[(name, part)] = path
        timings.append(dict(model=name, cache_seconds=time.perf_counter() - start))
        if name == selected:
            for part in ["tune", "audit"]:
                path = args.output / f"selected-tta-{part}.npy"
                cache_probabilities(model, loaders[part], device, path, tta=True)
                paths[("selected_tta", part)] = path
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for part in ["tune", "audit"]:
        destination = args.output / f"ensemble-{part}.npy"
        average_files([paths[(name, part)] for name in names], destination)
        paths[("ensemble", part)] = destination
    variants = [
        ("single", selected, [selected], False, 1),
        ("horizontal_tta", "selected_tta", [selected], True, 2),
        ("equal_mean_ensemble", "ensemble", names, False, 3),
    ]
    rows = []
    configs = {}
    for variant, cache, model_names, tta, cost in variants:
        tune = np.load(paths[(cache, "tune")], mmap_mode="r")
        audit = np.load(paths[(cache, "audit")], mmap_mode="r")
        thresholds, areas, trials = tune_postprocess(tune, truth["tune"])
        trials.to_csv(args.output / f"{variant}-postprocessing.csv", index=False)
        t = score_predictions(tune, truth["tune"], thresholds, areas)
        a = score_predictions(audit, truth["audit"], thresholds, areas)
        t.to_csv(args.output / f"{variant}-tune-metrics.csv", index=False)
        a.to_csv(args.output / f"{variant}-audit-metrics.csv", index=False)
        row = dict(
            variant=variant,
            model_names=model_names,
            tta=tta,
            forward_passes=cost,
            tune_assessment_score=float(t.assessment_score.mean()),
            audit_assessment_score=float(a.assessment_score.mean()),
            audit_macro_pixel_f1=float(a.pixel_f1.mean()),
            audit_pixel_accuracy=float(a.pixel_accuracy.mean()),
        )
        rows.append(row)
        configs[variant] = dict(
            row,
            thresholds=thresholds,
            min_areas=areas,
            model=selected if len(model_names) == 1 else "ensemble",
        )
        pd.DataFrame(rows).to_csv(args.output / "inference_comparison.csv", index=False)
        print(row, flush=True)
    best_score = max(row["tune_assessment_score"] for row in rows)
    eligible = [
        row for row in rows if row["tune_assessment_score"] >= best_score - 0.002
    ]
    chosen = min(eligible, key=lambda row: row["forward_passes"])
    decision = configs[chosen["variant"]]
    decision["selection_rule"] = (
        "highest tune score; prefer fewer forward passes within 0.002 of best"
    )
    (args.output / "final_selection.json").write_text(json.dumps(decision, indent=2))
    cache = dict((v[0], v[1]) for v in variants)[chosen["variant"]]
    audit = np.load(paths[(cache, "audit")], mmap_mode="r")
    pairs = pair_metrics(
        audit, truth["audit"], audit_ids, decision["thresholds"], decision["min_areas"]
    )
    pairs.to_csv(args.output / "audit-pair-metrics.csv", index=False)
    cases = error_figure(
        pairs,
        audit,
        truth["audit"],
        audit_ids,
        decision["thresholds"],
        decision["min_areas"],
        args.data,
        args.output / "error-analysis.png",
    )
    (args.output / "error-cases.json").write_text(json.dumps(cases, indent=2))
    pd.DataFrame(timings).to_csv(args.output / "inference-timings.csv", index=False)
    row_count = predict_test(
        decision, args.models, args.data, args.output, device, args.batch_size
    )
    run = wandb.init(
        entity="23f3000839-iit-madras-foundation",
        project="23f3000839-t32026",
        job_type="validation",
        name="final-inference-ablation",
        mode="offline",
        dir=str(args.output),
    )
    run.log({"inference_comparison": wandb.Table(dataframe=pd.DataFrame(rows))})
    run.summary.update(
        {
            "selected_variant": decision["variant"],
            "selected_model_names": decision["model_names"],
            "tune_assessment_score": decision["tune_assessment_score"],
            "audit_assessment_score": decision["audit_assessment_score"],
            "submission_rows": row_count,
        }
    )
    run.finish()
    for path in args.output.rglob("*.npy"):
        path.unlink()
    print("COMPLETED", json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
