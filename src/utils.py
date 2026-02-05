import yaml
import torch
from pathlib import Path


def load_config(path):
    """
    Load configuration from a YAML file.
    Example:
        cfg = load_config("config.yaml")

    Use case example:
        batch_size = cfg["training"]["batch_size"]
    
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

        # This section is to ensure that if the individual paths are not set, 
        # they will be derived from the data_root
        root = Path(cfg["paths"]["data_root"])
        for split in ["train_data", "val_data", "test_data"]:
            if cfg["paths"].get(split) is None:
                cfg["paths"][split] = str(root / split.split("_")[0])
    return cfg

def get_device(cfg):
    """
    Determine the computation device based on configuration.
    Exanple:
        device = get_device(cfg)
    """
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])

