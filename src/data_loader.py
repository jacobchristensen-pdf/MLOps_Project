# Finde filer
from pathlib import Path
from PIL import Image, ImageFile
import torch
import torchvision.transforms as T
import random
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allows loading of corrupted images without crashing
# Suppress PIL warnings a verify method has been added to handle these types of files
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


def get_files(directory):
    "Makes a sorted list with all file paths"
    directory = Path(directory)             # converts to Path object for easy file handling
    return sorted([
        file_path
        for file_path in directory.iterdir()    # Iterates through all items in the directory
        if file_path.is_file()                  # Only include files, ignore subdirectories
    ])


def load_image(image_path):
    "Opens picture with PIL and converts it to RGB"
    try:
        img = Image.open(image_path)
        img.verify()  # Verifies that the image is not corrupted
        return Image.open(image_path).convert("RGB")  # Secures RGB 3-channel input
    except Exception as e:
        print(f"CORRUPTED: {image_path} - {e}")
        return None


def totensor(image_size: int):
    "Transform to tensor and resize"
    return T.Compose([
        T.Resize((image_size, image_size)),  # Same resize input for all images
        T.ToTensor()                         # Converts to PyTorch tensor
    ])


class CatsDogsLoader:
    def __init__(self, root_dir, image_size):
        root_dir = Path(root_dir)
        self.image_size = image_size

        # Loads file paths for each class
        cat_paths = get_files(root_dir / "Cat")
        dog_paths = get_files(root_dir / "Dog")

        print("\nScanning for corrupted images")

        # Filter out corrupted images
        self.cat_image_paths = [p for p in cat_paths if load_image(p) is not None]
        self.dog_image_paths = [p for p in dog_paths if load_image(p) is not None]

        # Log how many images were skipped
        skipped_cats = len(cat_paths) - len(self.cat_image_paths)
        skipped_dogs = len(dog_paths) - len(self.dog_image_paths)
        print("\n finished scanning:")
        print(f"   Valid cats: {len(self.cat_image_paths)}")
        print(f"   Valid dogs: {len(self.dog_image_paths)}")
        print(f"   Skipped cats: {skipped_cats}")
        print(f"   Skipped dogs: {skipped_dogs}\n")

        # Transform applied to all images
        self.transform = totensor(image_size)

        # Compiled list of (image_path, class_label)
        self.samples = (
            [(image_path, 0) for image_path in self.cat_image_paths] +  # cat = 0
            [(image_path, 1) for image_path in self.dog_image_paths]    # dog = 1
        )

    def sample(self):
        # Random sample (used for debugging and visualization)
        class_label = random.randint(0, 1)
        image_path = random.choice(
            self.cat_image_paths if class_label == 0 else self.dog_image_paths
        )

        image = load_image(image_path)
        if image is None:
            return None  # Skip corrupted images

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(class_label, dtype=torch.long)

        return image_tensor, label_tensor, image_path

    def __len__(self):
        return len(self.samples)  # Amount of samples in the dataset

    def __getitem__(self, index):
        image_path, class_label = self.samples[index]  # Choose sample via index

        image = load_image(image_path)
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(class_label, dtype=torch.long)

        return image_tensor, label_tensor  # Returns input and label
