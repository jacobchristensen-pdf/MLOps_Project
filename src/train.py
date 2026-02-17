from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data_loader import CatsDogsLoader
from .model import build_model
from .utils import get_device, load_config


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for image_tensors, label_tensors in data_loader:
        image_tensors = image_tensors.to(device)
        label_tensors = label_tensors.to(device)

        optimizer.zero_grad()
        logits = model(image_tensors)
        loss = loss_fn(logits, label_tensors)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


@torch.no_grad()
def validate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for image_tensors, label_tensors in data_loader:
        image_tensors = image_tensors.to(device)
        label_tensors = label_tensors.to(device)

        logits = model(image_tensors)
        loss = loss_fn(logits, label_tensors)
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        total += label_tensors.size(0)
        correct += (predicted == label_tensors).sum().item()

    accuracy = correct / total
    return total_loss / len(data_loader), accuracy


def main():
    config = load_config("configurations/base.yaml")
    device = get_device(config)
    print("Device:", device)

    # ========== MLflow: Kun 3 linjer! ==========
    mlflow.set_tracking_uri("http://172.24.198.42:5000")
    mlflow.set_experiment("CatsDogs")

    with mlflow.start_run():

        # ========== Log vigtigste parameters ==========
        mlflow.log_param("learning_rate", config["training"]["learning_rate"])
        mlflow.log_param("epochs", config["training"]["epochs"])
        mlflow.log_param("batch_size", config["training"]["batch_size"])

        # Create datasets
        train_dataset = CatsDogsLoader(
            config["paths"]["train_data"], config["dataset"]["image_size"]
        )
        val_dataset = CatsDogsLoader(
            config["paths"]["val_data"], config["dataset"]["image_size"]
        )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["misc"]["workers"],
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
        )

        # Initialize model
        model = build_model(num_classes=2).to(device)

        # Loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        # Best model tracking
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        best_model_path = Path(config["paths"]["out_dir"]) / "best.pt"
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Training loop
        for epoch in range(1, config["training"]["epochs"] + 1):
            train_loss = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device
            )
            val_loss, val_accuracy = validate(model, val_loader, loss_fn, device)

            # ========== Log metrics ==========
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(
                f"Epoch {epoch}/{config['training']['epochs']} | "
                f"train loss={train_loss:.4f} | "
                f"val loss={val_loss:.4f} | "
                f"val acc={val_accuracy:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Saved new best model (val_loss={val_loss:.4f})")

        # ========== Gem model i MLflow ==========
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)
        mlflow.pytorch.log_model(model, "model")

        print("\n✅ Done! Check MLflow: http://172.24.198.42:5000")


if __name__ == "__main__":
    main()
