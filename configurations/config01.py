
from pathlib import Path
import torch

"""
This file contains configuration parameters for the initial setup.

When creating a new configuration, please copy this file and modify the parameters as needed.

Method to use multiple configurations:
1. Import the desired configuration file in your main script.
2. Access the parameters using the imported module.

Example:
import configurations.config01 as config  <-- change config01 to your desired configuration file
print(config.BATCH_SIZE)

"""

# Paths
TRAIN_DIR = r"C:\Users\alext\Downloads\down.micr.com_down_3_E_1_3E1C-ECDB-4869-83t5dL0AqEqZkh827kQD8ImFN3e1ro0VHHaobmSQAzSvk\PetImages\train"
VAL_DIR   = r"C:\Users\alext\Downloads\down.micr.com_down_3_E_1_3E1C-ECDB-4869-83t5dL0AqEqZkh827kQD8ImFN3e1ro0VHHaobmSQAzSvk\PetImages\val"
OUT_DIR   = "runs"


# Misc
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS     = 5
LR         = 1e-4
NUM_WORKERS = 4


WEIGHT_DECAY   = 0.0