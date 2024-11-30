import os
import torch
import pandas as pd
from PIL import Image

class PascalVOCDataset(torch.utils.data.Dataset):
	def __init__(self, csv_path, images_dir, labels_dir, transforms = None):
		self.csv = pd.read_csv(csv_path)
		self.images_dir = images_dir
		self.labels_dir = labels_dir
		self.transforms = transforms

	def __len__(self):
		return len(self.csv)

	def __getitem__(self, idx):
		row = self.csv.iloc[idx]
		image_path = os.path.join(self.images_dir, row['image_id'])
		image = Image.open(image_path).convert('RGB')
		boxes = eval(row['bboxes'])
		labels = eval(row['labels'])
		if self.transforms:
			image = self.transforms(image)

		boxes = torch.tensor(boxes, dtype = torch.float32)
		labels = torch.tensor(labels, dtype = torch.float32)
		return image, {"boxes": boxes, "labels": labels}



