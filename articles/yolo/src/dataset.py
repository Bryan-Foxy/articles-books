# # Classes
# VOC classes
# names:
#   0: aeroplane
#   1: bicycle
#   2: bird
#   3: boat
#   4: bottle
#   5: bus
#   6: car
#   7: cat
#   8: chair
#   9: cow
#   10: diningtable
#   11: dog
#   12: horse
#   13: motorbike
#   14: person
#   15: pottedplant
#   16: sheep
#   17: sofa
#   18: train
#   19: tvmonitor

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

class PascalVOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_dir, labels_dir, transform=None):
        self.data = pd.read_csv(csv_path, header=None, names=['image_id', 'label_id'])
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_file = row["image_id"]
        label_file = row["label_id"]

        image_path = os.path.join(self.images_dir, image_file)
        label_path = os.path.join(self.labels_dir, label_file)
        image = np.array(Image.open(image_path).convert("RGB"))

        boxes, labels = [], []
        with open(label_path, "r") as f:
            for line in f:
                data = line.strip().split()
                labels.append(int(data[0]))
                boxes.append(list(map(float, data[1:])))  # YOLO: x_center, y_center, width, height

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.transform:
            augmented = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = augmented["image"]
            boxes = torch.tensor(augmented["bboxes"], dtype=torch.float32)
            labels = torch.tensor(augmented["class_labels"], dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def visualize_image_with_bboxes(image, boxes, labels, class_names=None):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)

        for box, label in zip(boxes, labels):
            x_center, y_center, width, height = box
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)

            rect = patches.Rectangle(
                (x_min * image.shape[1], y_min * image.shape[0]),
                width * image.shape[1],
                height * image.shape[0],
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
            label_text = str(label) if class_names is None else class_names[label]
            ax.text(
                x_min * image.shape[1],
                y_min * image.shape[0] - 5,
                label_text,
                color='red',
                fontsize=12,
                backgroundcolor="white"
            )

        plt.axis("off")
        plt.savefig('saves/example_in_dataset.png')
        plt.show()
    
    dset = PascalVOCDataset("/Users/armandbryan/Downloads/archive/train.csv", "/Users/armandbryan/Downloads/archive/images", "/Users/armandbryan/Downloads/archive/labels")
    sample_image, sample_target = dset[2]
    visualize_image_with_bboxes(sample_image, sample_target["boxes"], sample_target["labels"])