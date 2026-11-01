"""Train a baseline using the supplied Milestone 1 split, loss and metric."""
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb
from data import SurfaceDataset, CLASSES
from models import SmallUNet
from metrics import segmentation_loss, validation_loss, threshold_sweep

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/baseline'))
    parser.add_argument('--model', choices=['scratch', 'resnet34'], default='scratch')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    seed, batch_size = 42, 8
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(args.data / 'train.csv', keep_default_na=False)
    annotations['mask_rle'] = annotations.mask_rle.str.strip()
    table = annotations.pivot(index='image_id', columns='defect_type', values='mask_rle').reindex(columns=CLASSES).sort_index()
    presence = table.ne('')
    combos = presence.apply(lambda row: '+'.join(c for c in CLASSES if row[c]) or 'clean', axis=1)
    train_ids, val_ids = train_test_split(table.index.to_numpy(), test_size=0.2, random_state=seed, stratify=combos)
    assert not set(train_ids) & set(val_ids)
    split = pd.DataFrame({'label_combo': combos, 'partition':'train'})
    split.loc[val_ids, 'partition'] = 'validation'
    split.to_csv(args.output / 'holdout.csv', index_label='image_id')
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=device.type=='cuda')
    train_loader = DataLoader(SurfaceDataset(train_ids, table, args.data), shuffle=True, generator=torch.Generator().manual_seed(seed), **loader_args)
    val_loader = DataLoader(SurfaceDataset(val_ids, table, args.data), shuffle=False, **loader_args)
    if args.model == 'scratch':
        model = SmallUNet().to(device)
    else:
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=4, activation=None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    run = wandb.init(entity='23f3000839-iit-madras-foundation', project='23f3000839-t32026', name=f'{args.model}-seed42', job_type='training', config={'model':args.model,'seed':seed,'epochs':args.epochs,'batch_size':batch_size,'height':128,'width':800,'train_images':len(train_ids),'validation_images':len(val_ids),'loss':'0.5 BCE + 0.5 positive Dice','augmentation':'none'}, dir=str(args.output))
    history, best_loss = [], float('inf')
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum, samples = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(images)
            samples += len(images)
        val_loss = validation_loss(model, val_loader)
        row = {'epoch':epoch,'train_loss':loss_sum/samples,'val_loss':val_loss}
        history.append(row)
        run.log(row)
        pd.DataFrame(history).to_csv(args.output/'training_history.csv',index=False)
        print(row,flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),args.output/'best.pt')
    model.load_state_dict(torch.load(args.output/'best.pt',map_location=device,weights_only=True))
    scores = threshold_sweep(model,val_loader,[0.3,0.5,0.7,0.9])
    scores.to_csv(args.output/'threshold_sweep.csv',index=False)
    run.log({'threshold_results':wandb.Table(dataframe=scores)})
    best = scores.loc[scores.assessment_score.idxmax()]
    run.summary.update({'best_assessment_score':float(best.assessment_score),'selected_threshold':float(best.threshold),'best_val_loss':best_loss})
    (args.output/'wandb_url.txt').write_text(run.url)
    run.finish()
    print(scores.to_string(index=False))

if __name__ == '__main__':
    main()
