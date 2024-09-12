from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

DEFAULT_RESIZED_IMAGE_SIZE = 128
CSV_DATA_SET_PATH = "../image_paths_and_classes.csv"


def get_transforms(grayscale: bool = False, resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE):
    """
    Returns the transforms that are applied to the images when they are loaded.
    Can be customized for grayscale or color images and resized dimensions.
    """
    if grayscale:
        image_transforms = transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Single channel for grayscale
            ]
        )
    else:
        image_transforms = transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # RGB channels
            ]
        )
    return image_transforms


def load_dataset(csv_path=CSV_DATA_SET_PATH):
    """
    Loads the dataset from a CSV file.
    Assumes the CSV contains two columns: 'image_path' and 'class'.
    """
    print("Current Working Directory:", os.getcwd())

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns or 'class' not in df.columns:
        raise ValueError("CSV must contain 'image_path' and 'class' columns")
    
    return df, df["image_path"].values, df["class"].values


class PokemonDataset(Dataset):
    """
    Custom dataset class for Pokémon images.
    Loads images and applies optional transformations.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Check if the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error opening image {image_path}: {e}")

        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


if __name__ == "__main__":
    # Example usage
    dataset, image_paths, labels = load_dataset()
    transform = get_transforms()
    pokemonDS = PokemonDataset(image_paths, labels, transform)

