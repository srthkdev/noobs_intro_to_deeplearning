from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
HEIGHT, WIDTH = 128, 800

def decode_rle(rle):
    flat = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    if not str(rle).strip():
        return flat.reshape(HEIGHT, WIDTH, order="F")
    tokens = np.asarray(str(rle).split(), dtype=np.int64)
    if len(tokens) % 2:
        raise ValueError("RLE needs start/length pairs")
    starts, lengths = tokens[0::2] - 1, tokens[1::2]
    ends = starts + lengths
    if np.any(starts < 0) or np.any(starts >= flat.size) or np.any(lengths <= 0):
        raise ValueError("Invalid RLE start or length")
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError("Unsorted or overlapping runs in one mask")
    for start, end in zip(starts, np.minimum(ends, flat.size)):
        flat[start:end] = 1.0
    return flat.reshape(HEIGHT, WIDTH, order="F")

class SurfaceDataset(Dataset):
    def __init__(self, image_ids, rle_table, data_root):
        self.image_ids = list(image_ids)
        self.rle_table = rle_table
        self.data_root = Path(data_root)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        filename = image_id if image_id.endswith(".jpg") else image_id + ".jpg"
        with Image.open(self.data_root / "train" / filename) as handle:
            image = np.asarray(handle.convert("L"), dtype=np.float32) / 255.0
        if image.shape != (HEIGHT, WIDTH):
            raise ValueError(f"Wrong image shape for {image_id}: {image.shape}")
        # Normalize independently per image; train and validation use the same rule.
        image = (image - image.mean()) / max(float(image.std()), 0.05)
        masks = np.stack([decode_rle(self.rle_table.loc[image_id, c]) for c in CLASSES])
        return torch.from_numpy(image[None].copy()), torch.from_numpy(masks.copy())
