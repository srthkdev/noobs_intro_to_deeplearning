import torch
from torch import nn
import pandas as pd
from tqdm.auto import tqdm
from data import HEIGHT, WIDTH
bce_loss = nn.BCEWithLogitsLoss()

def segmentation_loss(logits, targets):
    bce = bce_loss(logits, targets)
    probs = logits.sigmoid()
    intersection = (probs * targets).sum(dim=(-2, -1))
    denominator = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    soft_dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    positive = (targets.sum(dim=(-2, -1)) > 0).to(logits.dtype)
    dice_loss = ((1 - soft_dice) * positive).sum() / positive.sum().clamp_min(1)
    return 0.5 * bce + 0.5 * dice_loss

@torch.no_grad()
def validation_loss(model, loader):
    device = next(model.parameters()).device
    model.eval()
    total, samples = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        loss = segmentation_loss(model(images), targets)
        total += loss.item() * len(images)
        samples += len(images)
    return total / samples

@torch.no_grad()
def threshold_sweep(model, loader, thresholds):
    device = next(model.parameters()).device
    model.eval()
    totals = {tau: dict(pairs=0, positive=0, empty=0, dice_sum=0.0,
                       fg_dice_sum=0.0, fp_fraction_sum=0.0, correct_empty=0)
              for tau in thresholds}
    for images, targets in tqdm(loader, desc="Validation threshold sweep"):
        probs = model(images.to(device)).sigmoid()
        truth = targets.to(device) > 0.5
        gt_pixels = truth.sum(dim=(-2, -1))
        positive, empty = gt_pixels > 0, gt_pixels == 0
        for tau in thresholds:
            prediction = probs >= tau  # Raw probabilities; no component/area filtering.
            pred_pixels = prediction.sum(dim=(-2, -1))
            intersection = (prediction & truth).sum(dim=(-2, -1))
            denominator = pred_pixels + gt_pixels
            dice = torch.where(
                denominator == 0, torch.ones_like(denominator, dtype=torch.float32),
                2.0 * intersection / denominator.clamp_min(1),
            )
            total = totals[tau]
            total["pairs"] += truth.shape[0] * truth.shape[1]
            total["positive"] += int(positive.sum().item())
            total["empty"] += int(empty.sum().item())
            total["dice_sum"] += dice.sum().item()
            total["fg_dice_sum"] += dice[positive].sum().item()
            total["fp_fraction_sum"] += (pred_pixels[empty].float() / (HEIGHT * WIDTH)).sum().item()
            total["correct_empty"] += int((empty & (pred_pixels == 0)).sum().item())
    rows = []
    for tau, total in totals.items():
        if not total["positive"] or not total["empty"]:
            raise ValueError("The holdout needs both positive and empty image–class pairs")
        fg = total["fg_dice_sum"] / total["positive"]
        fp_fraction = total["fp_fraction_sum"] / total["empty"]
        empty_accuracy = total["correct_empty"] / total["empty"]
        rows.append({
            "threshold": tau,
            "naive_dice": total["dice_sum"] / total["pairs"],
            "foreground_dice": fg,
            "fp_pixel_fraction": fp_fraction,
            "fp_pair_rate": 1 - empty_accuracy,
            "false_positive_score": 1 - fp_fraction,
            "empty_accuracy": empty_accuracy,
            "assessment_score": 0.70 * fg + 0.20 * (1 - fp_fraction) + 0.10 * empty_accuracy,
            "holdout_empty_pair_fraction": total["empty"] / total["pairs"],
        })
    return pd.DataFrame(rows).sort_values("threshold")

