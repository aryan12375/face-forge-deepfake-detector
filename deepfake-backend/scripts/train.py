"""
train.py
========
Fine-tunes EfficientNet-B4 on a deepfake dataset.

Expects this directory layout:
    data/
      train/
        real/   *.jpg / *.png
        fake/   *.jpg / *.png
      val/
        real/
        fake/

Usage:
    python scripts/train.py \
        --data_dir ./data \
        --epochs 20 \
        --batch_size 16 \
        --lr 3e-4 \
        --output_dir ./checkpoints

The saved checkpoint is compatible with ModelService — just point
ModelService.load() at the weights file.
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Augmentation pipelines ────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((420, 420)),
    transforms.RandomCrop(380),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    # Simulate social-media JPEG compression artefacts
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # Random JPEG-quality degradation (simulate compression robustness)
    transforms.RandomApply([
        transforms.Lambda(lambda t: t + 0.01 * torch.randn_like(t))
    ], p=0.3),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(device: torch.device) -> nn.Module:
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 2),
    )
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — stabilises fine-tuning
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 20 == 0:
            logger.info(
                f"Epoch {epoch} | step {batch_idx}/{len(loader)} "
                f"| loss {loss.item():.4f} | acc {100 * correct / total:.1f}%"
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument("--data_dir",    type=str, default="./data")
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--output_dir",  type=str, default="./checkpoints")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze EfficientNet features, train head only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Training on: {device}")

    # Data
    train_dataset = datasets.ImageFolder(
        root=Path(args.data_dir) / "train",
        transform=TRAIN_TRANSFORM,
    )
    val_dataset = datasets.ImageFolder(
        root=Path(args.data_dir) / "val",
        transform=VAL_TRANSFORM,
    )

    logger.info(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    logger.info(f"Classes: {train_dataset.class_to_idx}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Model
    model = build_model(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen — training head only.")

    # Class weights to handle imbalanced datasets
    class_counts = [len(list((Path(args.data_dir) / "train" / c).glob("*")))
                    for c in ["fake", "real"]]
    total = sum(class_counts)
    weights = torch.tensor([total / (2 * c) for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"── Epoch {epoch}/{args.epochs} "
            f"| train_loss {train_loss:.4f} "
            f"| val_loss {val_loss:.4f} "
            f"| val_acc {val_acc:.2f}% "
            f"| {elapsed:.0f}s"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, ckpt_path)
            logger.info(f"✓ New best saved → {ckpt_path} (val_acc={val_acc:.2f}%)")

        # Save latest checkpoint every 5 epochs
        if epoch % 5 == 0:
            latest_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), latest_path)

    logger.info(f"Training complete. Best val_acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
