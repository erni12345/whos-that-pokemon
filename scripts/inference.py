import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import argparse

# Import the model architecture
from src.models.CNN import PokemonCNN

# Function to load the class index to class name mapping
def load_class_mapping(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Class mapping CSV not found at {csv_path}")
    
    class_mapping_df = pd.read_csv(csv_path)
    if 'class_index' not in class_mapping_df.columns or 'class_name' not in class_mapping_df.columns:
        raise ValueError("CSV must contain 'class_index' and 'class_name' columns")

    class_mapping = dict(zip(class_mapping_df['class_index'], class_mapping_df['class_name']))
    return class_mapping

# Function to preprocess the input image
def preprocess_image(image_path, resize_size=128):
    # Define the transformations: resizing and normalization
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Open the image, apply the transformations
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Function to load the model
def load_model(model_path, num_classes):
    model = PokemonCNN(num_pokemon=num_classes)
    checkpoint = torch.load(model_path)  # Load the entire checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load only the model state_dict
    model.eval()  # Set model to evaluation mode
    return model


# Function to infer the class from an image
def infer(image_path, model, class_mapping, device):
    # Preprocess the image
    image = preprocess_image(image_path).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)

    # Get the predicted class index
    predicted_idx = predicted_idx.item()
    
    # Map the predicted class index to the Pokémon name
    predicted_class_name = class_mapping.get(predicted_idx, "Unknown")
    
    return predicted_class_name

def main(image_path, model_path, class_mapping_csv):
    # Check if the files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the class index to class name mapping
    class_mapping = load_class_mapping(class_mapping_csv)
    num_classes = len(class_mapping)

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, num_classes).to(device)

    # Perform inference and get the predicted Pokémon name
    predicted_pokemon = infer(image_path, model, class_mapping, device)
    print(f"Predicted Pokémon: {predicted_pokemon}")

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Infer the Pokémon from an image using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file.")
    parser.add_argument("--class_mapping_csv", type=str, default="src/class_mapping.csv",
                        help="Path to the CSV file containing class index to class name mapping.")
    
    args = parser.parse_args()
    
    # Run the main inference process
    main(args.image_path, args.model_path, args.class_mapping_csv)

