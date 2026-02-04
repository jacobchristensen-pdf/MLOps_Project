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



############################ KODE BRUGT TIL AT SPLITTE DATASÆTTET I 80% TRAIN OG 20% VAL ############################

import os
import random
import shutil

# --- Konfiguration ---
SOURCE_DIR = r"C:\Users\alext\Downloads\down.micr.com_down_3_E_1_3E1C-ECDB-4869-83t5dL0AqEqZkh827kQD8ImFN3e1ro0VHHaobmSQAzSvk\PetImages"
VAL_DIR = "val"
SPLIT_RATIO = 0.20
CLASSES = ["Dog", "Cat"]

random.seed(42)  # for reproducerbarhed

for cls in CLASSES:
    src_cls_dir = os.path.join(SOURCE_DIR, cls)
    val_cls_dir = os.path.join(VAL_DIR, cls)

    os.makedirs(val_cls_dir, exist_ok=True)

    images = [
        f for f in os.listdir(src_cls_dir)
        if os.path.isfile(os.path.join(src_cls_dir, f))
    ]

    num_val = int(len(images) * SPLIT_RATIO)
    val_images = random.sample(images, num_val)

    for img in val_images:
        src_path = os.path.join(src_cls_dir, img)
        dst_path = os.path.join(val_cls_dir, img)

        shutil.copy(src_path, dst_path)
        # brug denne i stedet hvis du vil FLYTTE:
        # shutil.move(src_path, dst_path)

    print(f"{cls}: {num_val} billeder flyttet til val/")

############################ KODE BRUGT TIL AT SPLITTE DATASÆTTET I 80% TRAIN OG 20% VAL ############################