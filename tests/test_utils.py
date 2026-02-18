import yaml
import torch
from pathlib import Path

from src.utils import load_config, get_device

# If only data root is given, make sure train/val/test split is done
def test_load_config_train_val_test_split(tmp_path):
    cfg_dict = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "train_data": None,
            "val_data": None,
            "test_data": None,
        },
        "device": "cpu",
    }

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg_dict))

    cfg = load_config(config_path)

    root = Path(cfg_dict["paths"]["data_root"])
    assert cfg["paths"]["train_data"] == str(root / "train")
    assert cfg["paths"]["val_data"] == str(root / "val")
    assert cfg["paths"]["test_data"] == str(root / "test")

# Makes sure that, if paths to train/val/test is given -> Utils does not overwrite
def test_load_config_dont_overwrite_existing_paths(tmp_path):
    cfg_dict = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "train_data": "/custom/train",
            "val_data": "/custom/val",
            "test_data": "/custom/test",
        },
        "device": "cpu",
    }

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg_dict))

    cfg = load_config(config_path)

    assert cfg["paths"]["train_data"] == "/custom/train"
    assert cfg["paths"]["val_data"] == "/custom/val"
    assert cfg["paths"]["test_data"] == "/custom/test"

# If no GPU avaliable make sure it runs on CPU
def test_get_device_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = {"device": "auto"}
    device = get_device(cfg)
    assert device.type == "cpu"

# If told to run on CPU, check if it does so
def test_get_device_explicit_cpu():
    cfg = {"device": "cpu"}
    device = get_device(cfg)
    assert device.type == "cpu"