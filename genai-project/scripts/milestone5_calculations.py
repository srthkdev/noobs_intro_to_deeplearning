"""Reproduce the nine current GenAI M5 answers; no model training is needed."""

from pathlib import Path
import json, math, argparse
from collections import Counter
import numpy as np
import pandas as pd
from scipy import ndimage

CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
H, W = 128, 800


def decode_rle(value):
    a = np.zeros(H * W, np.uint8)
    if str(value).strip():
        for start, length in np.asarray(str(value).split(), dtype=np.int64).reshape(
            -1, 2
        ):
            assert start >= 1 and length > 0
            a[start - 1 : min(start - 1 + length, H * W)] = 1
    return a.reshape(H, W, order="F")


def dice(pred, truth):
    return (
        float(2 * np.logical_and(pred, truth).sum() / (pred.sum() + truth.sum()))
        if pred.sum() + truth.sum()
        else 1.0
    )


def official(fg, fp, empty):
    return 0.7 * fg + 0.2 * fp + 0.1 * empty


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    ans = {}
    details = {}
    logits = np.array([-2.0, -0.9, -0.5, 0.0, 0.8, 2.2])
    truth = np.array([0, 1, 1, 1, 1, 0], bool)
    p = 1 / (1 + np.exp(-logits))
    values = [
        math.log(0.3 / 0.7),
        int((p > 0.5).sum()),
        int((p >= 0.5).sum()),
        dice(p > 0.3, truth),
        dice(p > 0.5, truth),
    ]
    ans["Q1"] = (
        f"{values[0]:.4f}, {values[1]}, {values[2]}, {values[3]:.4f}, {values[4]:.4f}"
    )
    details["Q1"] = {"probabilities": p.tolist(), "computed": values}
    configs = {
        "X": (0.62, 0.9950, 0.40),
        "Y": (0.58, 0.9990, 0.85),
        "Z": (0.45, 0.9998, 0.97),
    }
    scores = {k: official(*v) for k, v in configs.items()}
    ans["Q2"] = (
        ", ".join(f"{scores[k]:.4f}" for k in ["X", "Y", "Z"])
        + ", Y, FalsePositiveScore and EmptyAccuracy"
    )
    details["Q2"] = scores
    mask = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 1],
        ],
        np.uint8,
    )
    lab, n8 = ndimage.label(mask, structure=np.ones((3, 3)))
    _, n4 = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    kept = [int(sizes[sizes >= n].sum()) for n in [5, 6]]
    ans["Q3"] = f"{n8}, {n4}, {kept[0]}, {kept[1]}"
    details["Q3"] = {
        "mask": mask.tolist(),
        "8_connected_areas": sizes.tolist(),
        "4_connected_components": int(n4),
    }
    fg_before = (12 * 0.7 + 8 * 0.45) / 20
    fp_before = 1 - (20 * 20 + 10 * 2000) / (80 * H * W)
    ea_before = 50 / 80
    fg_after = (12 * 0.7 + 7 * 0.45) / 20
    fp_after = 1 - 10 * 2000 / (80 * H * W)
    ea_after = 70 / 80
    before = official(fg_before, fp_before, ea_before)
    after = official(fg_after, fp_after, ea_after)
    ans["Q4"] = (
        f"{fg_after:.4f}, {fp_after:.4f}, {ea_after:.4f}, {before:.4f}, {after:.4f}"
    )
    details["Q4"] = {
        "before": dict(fg=fg_before, fp=fp_before, empty=ea_before, score=before),
        "after": dict(fg=fg_after, fp=fp_after, empty=ea_after, score=after),
    }
    raw = pd.read_csv(root / "train.csv", keep_default_na=False)
    wiped = Counter()
    positive = 0
    for row in raw.itertuples(index=False):
        if not str(row.mask_rle).strip():
            continue
        positive += 1
        lab, n = ndimage.label(
            decode_rle(row.mask_rle), structure=np.ones((3, 3), np.uint8)
        )
        area = np.bincount(lab.ravel())[1:]
        if len(area) == 0 or area.max() < 100:
            wiped[row.defect_type] += 1
    worst = max(CLASSES, key=lambda c: wiped[c])
    ans["Q5"] = f"{sum(wiped.values())}, {worst}"
    details["Q5"] = dict(
        full_csv_rows=len(raw),
        unique_images=int(raw.image_id.nunique()),
        positive_pairs=positive,
        wiped_pairs=sum(wiped.values()),
        wiped_by_class={c: wiped[c] for c in CLASSES},
        connectivity=8,
        minimum_area=100,
    )
    original = np.array([[0.9, 0.6], [0.4, 0.2]])
    hraw = np.array([[0.5, 0.8], [0.3, 0.4]])
    vraw = np.array([[0.1, 0.5], [0.7, 0.9]])
    mean = (original + np.flip(hraw, axis=1) + np.flip(vraw, axis=0)) / 3
    bug = (original + hraw + vraw) / 3
    ans["Q6"] = f"3, 2, {mean[0,1]:.4f}, {int((mean>.5).sum())}, {int((bug>.5).sum())}"
    details["Q6"] = {"correct_average": mean.tolist(), "buggy_average": bug.tolist()}
    probs = np.array(
        [
            [0.9, 0.4, 0.45],
            [0.55, 0.55, 0.2],
            [0.8, 0.7, 0.9],
            [0.1, 0.2, 0.3],
            [0.6, 0.45, 0.55],
        ]
    )
    y = np.array([1, 0, 1, 0, 1], bool)
    soft = probs.mean(axis=1) > 0.5
    hard = (probs > 0.5).sum(axis=1) >= 2
    weighted = float(probs[0] @ np.array([0.5, 0.3, 0.2]))
    probavg = (1 / (1 + math.exp(-1)) + 1 / (1 + math.exp(3))) / 2
    logitavg = 1 / (1 + math.exp(1))
    ans["Q7"] = (
        f"{dice(soft,y):.4f}, {dice(hard,y):.4f}, {weighted:.4f}, {probavg:.4f}, {logitavg:.4f}, prob_avg"
    )
    details["Q7"] = {
        "soft_mask": soft.astype(int).tolist(),
        "hard_mask": hard.astype(int).tolist(),
        "weighted_p1": weighted,
        "probability_average": probavg,
        "sigmoid_average_logit": logitavg,
    }
    maps = 5000 * 4 * H * W
    gb32 = maps * 4 / 1e9
    gb16 = maps * 2 / 1e9
    gb8 = maps / 1e9
    passes_a = 5000 * 3 * 3
    passes_b = 5000 * 2 * 2
    minutes_a = passes_a * 0.012 / 60
    minutes_b = passes_b * 0.012 / 60
    max3 = math.floor((30 - 8) * 60 / (5000 * 3 * 0.012))
    max2 = math.floor((30 - 8) * 60 / (5000 * 2 * 0.012))
    ans["Q8"] = (
        f"{gb32:.3f}, {gb16:.3f}, {gb8:.3f}, {9*gb32:.3f}, {4*gb32:.3f}, neither, {gb32:.3f}, {passes_a}, {passes_b}, {minutes_a:.1f}, {minutes_b:.1f}, {max3}, {max2}"
    )
    details["Q8"] = {
        "map_values": maps,
        "one_float32_set_GB": gb32,
        "naive_A_GB": 9 * gb32,
        "naive_B_GB": 4 * gb32,
        "passes_A": passes_a,
        "passes_B": passes_b,
        "pure_model_minutes_A": minutes_a,
        "pure_model_minutes_B": minutes_b,
        "maximum_models_3view": max3,
        "maximum_models_2view": max2,
    }
    ans[
        "Q9"
    ] = """- I would keep the native 128 x 800 image geometry and apply exactly the preprocessing used to train each checkpoint. I would put every model in evaluation mode and use inference_mode, with AMP where it is supported. Changing normalization or channel order at inference would feed a different input distribution to the model.
- For each image batch and each checkpoint, I would run the original image and the chosen flips. On BCHW tensors, horizontal flip uses dimension 3 and vertical flip uses dimension 2. I would undo each flip on its predicted probability map before accumulating it. Otherwise, the average would mix different physical pixels and blur or move defects.
- I would average the aligned probabilities over views, then combine models with fixed weights that sum to one. Thresholding comes after those averages: an early hard threshold throws away confidence information and changes a soft ensemble into a voting scheme. I would then apply each class's chosen threshold with the same strict comparison used in validation, followed by its 8-connected minimum-area filter. Filtering before thresholding does not define the same binary connected components. Finally, I would encode each mask with 1-based, column-major RLE.
- To reduce holdout overfitting, I would select thresholds and minimum areas on a tuning split and evaluate that frozen choice once on a separate audit split. I would also keep the search small and check whether the chosen settings remain useful across folds or seeds. I would track ForegroundDice, FalsePositiveScore and EmptyAccuracy separately. The all-empty baseline can obtain a high naive Dice simply because most pairs are empty, so that metric alone could reward thresholds that erase real defects.
- I would write a completely empty mask as a single space in mask_rle, rather than a missing value. Before submitting, I would check the CSV against sample_submission for the exact row count, columns, image/class IDs, ordering and valid RLE bounds. I would also run the whole notebook in the submission environment and check wall time and peak memory. A batchwise float32 running sum avoids storing every model/view prediction; I would leave time for decoding, component filtering, RLE and file I/O instead of spending the full limit on model passes."""
    assert (
        values[1:3] == [2, 3]
        and abs(values[3] - 0.75) < 1e-12
        and abs(values[4] - 1 / 3) < 1e-12
    )
    assert (n8, n4, *kept) == (3, 6, 11, 11)
    assert int((mean > 0.5).sum()) == 2 and int((bug > 0.5).sum()) == 1
    assert max3 == 7 and max2 == 11
    (out / "answers.json").write_text(json.dumps(ans, indent=2))
    (out / "calculations.json").write_text(json.dumps(details, indent=2))
    print(json.dumps(ans, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.data or next(
        p.parent
        for p in Path("/kaggle/input").rglob("train.csv")
        if (p.parent / "train").is_dir()
    )
    main(root, a.output)
