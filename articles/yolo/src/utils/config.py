import torch
import pandas as pd

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