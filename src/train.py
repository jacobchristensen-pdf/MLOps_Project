from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import build_model
from .data_loader import CatsDogsLoader
from configurations import config01 as config



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()                       # Places the model in training mode
    total_loss = 0.0                    

    for x, y in loader:
        x = x.to(device)                # Moves input to CPU/GPU
        y = y.to(device)                # Moves labels to CPU/GPU

        optimizer.zero_grad()           # Resets the gradients
        logits = model(x)               # Forward pass
        loss = criterion(logits, y)     # Computes the loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Updates the model parameters

        total_loss += loss.item()       # Accumulates loss

    return total_loss / len(loader)     # Averages the loss over the epoch


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()                        # Evaluation mode (disables dropout, etc.)
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)               # Forward pass
        loss = criterion(logits, y)     # Calculates validation loss

        total_loss += loss.item()

    return total_loss / len(loader)     # Average validation loss


def main():
    device = config.DEVICE
    print("Device:", device)

    # Create dataset and data loaders for batching and shuffling
    train_ds = CatsDogsLoader(config.TRAIN_DIR, config.IMAGE_SIZE)
    val_ds   = CatsDogsLoader(config.VAL_DIR, config.IMAGE_SIZE)

    # DataLoaders to batching and shuffling
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

    # Initialize the model
    model = build_model(num_classes=2).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Save the best model based on validation loss
    best_val_loss = float("inf")
    best_path = Path(config.OUT_DIR) / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    # Training loop
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

        # Save model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved new best model (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
