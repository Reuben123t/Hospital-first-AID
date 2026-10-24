from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class BrainCNN(nn.Module):
    """Small CNN trained from scratch; deliberately no pretrained backbone."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(256, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train(train); total_loss = 0.0; y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images); loss = loss_fn(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().tolist()); y_pred.extend(logits.argmax(1).cpu().tolist())
    return total_loss / len(loader.dataset), y_true, y_pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=1000); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-minutes', type=float, default=None, help='Stop after this wall-clock time and keep the best checkpoint.')
    p.add_argument('--val-fraction', type=float, default=0.2); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-size', type=int, default=224); p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    args = p.parse_args(); seed_everything(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.data_dir.exists(): raise FileNotFoundError(args.data_dir)
    tfm = transforms.Compose([transforms.Grayscale(1), transforms.Resize((args.image_size, args.image_size)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(8), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    eval_tfm = transforms.Compose([transforms.Grayscale(1), transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    full = datasets.ImageFolder(args.data_dir, transform=tfm); n_val = max(1, int(len(full) * args.val_fraction)); n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)); val_ds.dataset = datasets.ImageFolder(args.data_dir, transform=eval_tfm)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0); val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = BrainCNN(len(full.classes)).to(device)
    counts = np.bincount([full.targets[i] for i in train_ds.indices], minlength=len(full.classes)); weights = len(train_ds) / (len(full.classes) * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device)); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_f1 = -1.0; deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
    for epoch in range(1, args.epochs + 1):
        if deadline and time.monotonic() >= deadline: break
        tr_loss, _, _ = run_epoch(model, train_loader, loss_fn, opt, device, True); va_loss, y, pred = run_epoch(model, val_loader, loss_fn, opt, device, False)
        f1 = f1_score(y, pred, average='macro', zero_division=0); bal = balanced_accuracy_score(y, pred)
        print(f'epoch={epoch:03d} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} balanced_acc={bal:.4f} macro_f1={f1:.4f}')
        if f1 > best_f1:
            best_f1 = f1; torch.save({'model': model.state_dict(), 'classes': full.classes, 'image_size': args.image_size}, args.output_dir / 'best_model.pt')
            report = {'classes': full.classes, 'balanced_accuracy': bal, 'macro_f1': f1, 'confusion_matrix': confusion_matrix(y, pred).tolist(), 'classification_report': classification_report(y, pred, target_names=full.classes, output_dict=True, zero_division=0), 'device': str(device)}
            (args.output_dir / 'validation_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if deadline and time.monotonic() >= deadline: break
    print(f'best checkpoint: {args.output_dir / "best_model.pt"}')


if __name__ == '__main__': main()
