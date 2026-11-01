"""Generate the four masks per test image from the scratch U-Net checkpoint."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from models import SmallUNet

CLASSES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']

def encode_rle(mask):
    flat = mask.astype(np.uint8).ravel(order='F')
    changes = np.flatnonzero(np.diff(np.r_[0, flat, 0])) + 1
    changes[1::2] -= changes[::2]
    # Kaggle's CSV validation rejects a null cell. A space denotes an empty mask.
    return ' '.join(map(str, changes)) or ' '

class TestDataset(Dataset):
    def __init__(self, root, ids):
        self.root, self.ids = Path(root), list(ids)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        image_id = self.ids[index]
        filename = image_id if image_id.endswith('.jpg') else image_id + '.jpg'
        with Image.open(self.root / 'test' / filename) as handle:
            image = np.asarray(handle.convert('L'), dtype=np.float32) / 255.0
        image = (image - image.mean()) / max(float(image.std()), 0.05)
        return torch.from_numpy(image[None].copy()), image_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(4)
    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    submission = pd.read_csv(args.data / 'sample_submission.csv', keep_default_na=False)
    loader = DataLoader(TestDataset(args.data, submission.image_id.unique()), batch_size=8, shuffle=False)
    predictions = {}
    with torch.no_grad():
        for batch, (images, ids) in enumerate(loader):
            masks = (model(images.to(device)).sigmoid() >= args.threshold).cpu().numpy()
            for image_id, channels in zip(ids, masks):
                for c, mask in zip(CLASSES, channels):
                    predictions[(image_id, c)] = encode_rle(mask)
            if batch % 100 == 0:
                print(f'Inference batch {batch}/{len(loader)}', flush=True)
    submission['mask_rle'] = [predictions[(r.image_id, r.defect_type)] for r in submission.itertuples()]
    assert len(submission) == len(predictions)
    assert not submission.isna().any().any()
    submission.to_csv(args.output, index=False)
    print(f'Saved {len(submission)} rows to {args.output}', flush=True)

if __name__ == '__main__':
    main()
