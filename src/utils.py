import yaml
import torch

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