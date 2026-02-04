
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
DATA = Path("")

# Dataset
IMAGE_SIZE = () # (H, W)

# Hyperparametre
BATCH_SIZE = int 
EPOCHS = int 
LEARNING_RATE = int
WEIGHT_DECAY   = 0.0

# Misc
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = int

