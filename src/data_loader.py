# Finde filer
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import random

# Vi skal have filhåndtering (stier).
# Skal finde alle billedfiler i en mappe
# Skal returnere en liste med filstierne
def get_files(path):
    "Makes list with all file paths"
    path = Path(path)
    return sorted([p for p in path.iterdir() if p.is_file()])



# Vi skal have indlæsning af et billede
# åben et billede med PIL og sæt farvemode (RGB)
# Output skal være farvebillede
def load_image(path):
    "Opens picture with PIL and converts it to RGB"
    return Image.open(path).convert("RGB")


def totensor(image_size: int):
    "Transform to tensor and resize"
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])

class CatsDogsLoader: # VI SKAL EVENTUELT KIKKE PÅ OM ANTALLET AF KLASSER SKAL VÆRE EN VARIABEL
    def __init__(self, root_dir, image_size):
        root_dir = Path(root_dir)
        self.cat_files = get_files(root_dir / "Cat")
        self.dog_files = get_files(root_dir / "Dog")
        self.transform = totensor(image_size)

        self.samples = [(p, 0) for p in self.cat_files] + [(p, 1) for p in self.dog_files]

    def sample(self):
        label = random.randint(0, 1)  # 0 for cat, 1 for dog
        path = random.choice(self.cat_files if label == 0 else self.dog_files)

        img = load_image(path)
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, path
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = load_image(path)
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y