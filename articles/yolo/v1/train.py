import time
import os
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from darknet import YOLOv1
from articles.yolo.dataset import VOCDataset
from articles.functions.map import map
from articles.yolo.config import *
from loss_function import LossFunctionYOLO

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = device
BATCH_SIZE = 16 
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "/Users/armandbryan/Downloads/archive/images"
LABEL_DIR = "/Users/armandbryan/Downloads/archive/labels"
SAVE_MODEL_PATH = "weights/yolov1_final.pth.tar"  # Path to save the model

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = LossFunctionYOLO(S=7, B=2, C=20)

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    start_loading_data = time.time()
    train_dataset = VOCDataset(
        "/Users/armandbryan/Downloads/archive/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "/Users/armandbryan/Downloads/archive/test.csv", 
        transform=transform, 
        img_dir=IMG_DIR, 
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    ending_load_data = time.time()
    time_loading_data = ending_load_data - start_loading_data
    print(f"Loading Data finished. ({time_loading_data:.3f}) seconds")

    start_model = time.time()
    for epoch in range(EPOCHS):
        print(f"EPOCH: [{epoch}|{EPOCHS}]")
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = map(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)

    end_model = time.time()
    duration_model = end_model - start_model
    print(f"The model takes ({duration_model:.3f}) minutes to run correctly!")

    # Save the model
    if not os.path.exists("weights"):
        os.makedirs("weights")
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH}")

if __name__ == "__main__":
    main()