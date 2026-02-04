from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import build_model
from .data_loader import CatsDogsLoader
from configurations import config01 as config



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = CatsDogsLoader(config.TRAIN_DIR, config.IMAGE_SIZE)
    val_ds   = CatsDogsLoader(config.VAL_DIR, config.IMAGE_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    model = build_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_val_loss = float("inf")
    best_path = Path(config.OUT_DIR) / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"train loss={train_loss:.4f} | "
            f"val loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved new best model (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
