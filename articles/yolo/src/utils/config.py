import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def get_device():
	""" Return device to computation for neural network"""
	if torch.backends.mps.is_available() and torch.backends.mps.is_built():
	    device = torch.device("mps")
	    return device
	elif torch.cuda.is_available():
	    device = torch.device("cuda")
	    return device
	else:
	    device = torch.device("cpu")
	    return device

def get_num_classes(csv_path):
    data = pd.read_csv(csv_path)
    all_labels = data['labels']
    unique_labels = set()
    for labels in all_labels:
        label_list = eval(labels)
        unique_labels.update(label_list)

    return len(unique_labels)

def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function to handle variable-sized bounding boxes and labels.
    
    Args:
        batch (List): List of (image, targets) tuples
    
    Returns:
        Tuple of (batched images, list of target dictionaries)
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Stack images
    images = torch.stack(images, 0)
    
    return images, targets

def plot_losses(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig('../saves/training_loss_curve.png')
    plt.show()