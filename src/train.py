"""
python train.py --mode letter
python train.py --mode word
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model import LetterMLP, WordGRU
from landmarks import TOTAL_FEATURES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LetterDataset(Dataset):
    def __init__(self, root: str):
        self.samples = []  # list of (filepath, label_idx)
        self.classes = sorted(os.listdir(root))
        for label_idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feats = np.load(path).astype(np.float32)
        return torch.from_numpy(feats), label


class WordDataset(Dataset):
    def __init__(self, root: str):
        self.samples = []
        self.classes = sorted(os.listdir(root))
        for label_idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = np.load(path).astype(np.float32)  # (seq_len, features)
        return torch.from_numpy(clip), label


def train_model(model, train_loader, val_loader, epochs, lr, save_path, class_weights=None):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE) if class_weights is not None else None
    )

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += x.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)

        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"train_loss={train_loss/train_total:.4f} "
              f"train_acc={train_correct/train_total:.3f} "
              f"val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"Best val acc: {best_val_acc:.3f}. Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["letter", "word"], required=True)
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--models-dir", default="../models")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    if args.mode == "letter":
        root = os.path.join(args.data_root, "letters")
        dataset = LetterDataset(root)
        model = LetterMLP(input_dim=TOTAL_FEATURES, num_classes=len(dataset.classes))
        save_path = os.path.join(args.models_dir, "letter_model.pt")
    else:
        root = os.path.join(args.data_root, "words")
        dataset = WordDataset(root)
        model = WordGRU(input_dim=TOTAL_FEATURES, num_classes=len(dataset.classes))
        save_path = os.path.join(args.models_dir, "word_model.pt")

    print(f"Classes ({len(dataset.classes)}): {dataset.classes}")
    print(f"Total samples: {len(dataset)}")

    #Manages the class imbalance so that the model isn't biased towards the words which have more video samples when compared to other words.
    class_weights = None
    if args.mode == "word":
        counts = np.zeros(len(dataset.classes))
        for _, label in dataset.samples:
            counts[label] += 1
        weights = 1.0 / np.maximum(counts, 1)
        weights = weights / weights.sum() * len(dataset.classes)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class counts: {dict(zip(dataset.classes, counts.astype(int)))}")

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    train_model(model, train_loader, val_loader, args.epochs, args.lr, save_path, class_weights)

    # Save label map so live_app.py can decode predictions
    label_map_path = os.path.join(args.models_dir, f"{args.mode}_labels.json")
    with open(label_map_path, "w") as f:
        json.dump(dataset.classes, f)
    print(f"Saved label map to {label_map_path}")


if __name__ == "__main__":
    main()