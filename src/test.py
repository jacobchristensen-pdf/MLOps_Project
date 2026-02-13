import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import CatsDogsLoader
import torch
from tqdm import tqdm

import utils
import model as m


if __name__ == "__main__":
    #Load config
    config = utils.load_config("configurations/base.yaml")
    device = utils.get_device(config)

    # Test config
    print(f"Testing config -> Device = {config['device']}")


    # Load trained model
    model = m.build_model()
    model.load_state_dict(torch.load("runs/best.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # evaluation mode

    test_ds = CatsDogsLoader(config["paths"]["test_data"], config["dataset"]["image_size"])


    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["misc"]["workers"]
    )


    # Initialize counters
    total = 0
    correct = 0
    tp = tn = fp = fn = 0

    # Evaluation loop
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(test_loader, desc="Testing batches")):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()
            tp += ((predicted == 1) & (y == 1)).sum().item()
            tn += ((predicted == 0) & (y == 0)).sum().item()
            fp += ((predicted == 1) & (y == 0)).sum().item()
            fn += ((predicted == 0) & (y == 1)).sum().item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nFinal Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
