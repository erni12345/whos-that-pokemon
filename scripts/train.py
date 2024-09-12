import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

# Import custom modules
from src.models.CNN import PokemonCNN
from src.datasets.pokemonDataset import PokemonDataset, get_transforms, load_dataset
from src.utils import save_checkpoint, load_checkpoint, accuracy

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_path=None):
    model = model.to(device)
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        loop = tqdm(train_loader, leave=True)

        # Training loop
        for i, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=running_loss/(i+1), accuracy=100 * correct / total)

        # Validation step
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save checkpoint if validation improves
        if val_accuracy > best_val_accuracy:
            print(f"Saving checkpoint for epoch {epoch + 1} with validation accuracy: {val_accuracy:.2f}%")
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            best_val_accuracy = val_accuracy
    
    print("Training complete!")


# Evaluation function for validation/testing
def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total



def main():
    # Hyperparameters and file paths
    num_epochs = 20
    learning_rate = 0.0005
    batch_size = 32
    num_classes = 150  # Number of Pokémon classes

    # Paths
    csv_train_path = 'data/processed/train.csv'
    csv_val_path = 'data/processed/test.csv'
    checkpoint_path = 'outputs/checkpoints/model.pth'

    # Data transformations
    transform = get_transforms()

    # Load datasets
    train_dataset, train_image_paths, train_labels = load_dataset(csv_train_path)
    val_dataset, val_image_paths, val_labels = load_dataset(csv_val_path)

    # Create Dataset objects
    pokemon_train_ds = PokemonDataset(train_image_paths, train_labels, transform=transform)
    pokemon_val_ds = PokemonDataset(val_image_paths, val_labels, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(pokemon_train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(pokemon_val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model and move it to the correct device (GPU or CPU)
    model = PokemonCNN(num_pokemon=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move the model to the device

    # Initialize the loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize the optimizer AFTER the model has been moved to the correct device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Optionally, load model from a checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        load_checkpoint(checkpoint_path, model, optimizer)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_path)

if __name__ == "__main__":
    main()

