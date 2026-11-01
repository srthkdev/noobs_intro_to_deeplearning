"""Load a trained final-pipeline checkpoint and predict consistent masks.

This module is shared by the Kaggle inference notebook and optional Gradio app.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from final_pipeline import CLASSES, H, W, make_model, remove_components, encode_rle


class Segmenter:
    def __init__(self, checkpoint, selection, device=None):
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        self.selection = json.loads(Path(selection).read_text())
        checkpoint = Path(checkpoint)
        names = self.selection.get("model_names", [self.selection["model"]])
        self.models = []
        for name in names:
            if len(names) > 1 and not checkpoint.is_dir():
                raise ValueError("An ensemble requires a directory of checkpoints")
            if checkpoint.is_dir():
                nested = checkpoint / name / "best.pt"
                path = nested if nested.is_file() else checkpoint / f"{name}.pt"
            else:
                path = checkpoint
            saved = torch.load(path, map_location=self.device, weights_only=False)
            if saved["architecture"] != name:
                raise ValueError("Checkpoint and selection architecture disagree")
            model = make_model(name, pretrained=False).to(self.device)
            model.load_state_dict(saved["state_dict"])
            model.eval()
            self.models.append(model)

    @torch.inference_mode()
    def predict_batch(self, images, tta=None):
        if tta is None:
            tta = self.selection.get("tta", False)
        arrays = []
        for image in images:
            if isinstance(image, (str, Path)):
                with Image.open(image) as f:
                    a = np.asarray(f.convert("L"), dtype=np.float32) / 255
            else:
                raw = np.asarray(image)
                integer_input = np.issubdtype(raw.dtype, np.integer)
                a = raw.astype(np.float32)
                if a.ndim == 3:
                    rgb = a if integer_input or a.max() > 1 else a * 255
                    a = (
                        np.asarray(
                            Image.fromarray(
                                np.rint(rgb).clip(0, 255).astype("uint8")
                            ).convert("L"),
                            dtype=np.float32,
                        )
                        / 255
                    )
                elif integer_input or a.max() > 1:
                    a = a / 255
            if a.shape != (H, W):
                raise ValueError(
                    f"Expected an original {H}x{W} strip; got {a.shape}. Resizing would distort the measurement geometry."
                )
            arrays.append((a - 0.5) / 0.25)
        x = torch.from_numpy(np.stack(arrays)[:, None]).to(self.device)
        probabilities = []
        for model in self.models:
            p = model(x).sigmoid()
            if tta:
                p = (p + torch.flip(model(torch.flip(x, [-1])).sigmoid(), [-1])) / 2
            # Validation stores each model's probabilities as float16 before averaging.
            probabilities.append(p.float().cpu().numpy().astype(np.float16))
        p = (
            sum(value.astype(np.float32) for value in probabilities)
            / len(probabilities)
        ).astype(np.float16)
        masks = np.empty_like(p, dtype=bool)
        for i in range(len(p)):
            for c in range(4):
                masks[i, c] = remove_components(
                    p[i, c] >= self.selection["thresholds"][c],
                    self.selection["min_areas"][c],
                )
        return masks, p

    def rles(self, images):
        masks, _ = self.predict_batch(images)
        return [
            {c: encode_rle(mask) for c, mask in zip(CLASSES, item)} for item in masks
        ]
