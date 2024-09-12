import torch

def save_checkpoint(model, optimizer, epoch, file_path):
    """
    Save the model and optimizer state to a checkpoint file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")


def load_checkpoint(file_path, model, optimizer=None):
    """
    Load the model and optimizer state from a checkpoint file.
    """
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {file_path}")
    return checkpoint['epoch']


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the model outputs and true labels.
    """
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)
