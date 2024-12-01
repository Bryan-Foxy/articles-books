import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
from tqdm import tqdm
from dataset import PascalVOCDataset
from src.models.yolo import YOLO
from src.utils.loss import Loss
from src.utils.data_augmentation import get_yolo_augmentation
from src.utils.config import get_device, plot_losses, collate_fn

def train(args):
    device = get_device()
    print(f"Using device: {device}")
    
    train_dset = PascalVOCDataset(
        csv_path = args.train_csv,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        transform = get_yolo_augmentation(),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn = collate_fn
    )

    nc = 20
    #nc = get_num_classes(args.train_csv)
    print(f"Number of classes: {nc}")

    yolo = YOLO(num_classes=nc, num_anchors=args.num_anchors).to(device)
    optimizer = torch.optim.SGD(yolo.parameters(), lr=args.lr)
    criterion = Loss(num_classes=nc).to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []

    for epoch in range(args.epochs):
        yolo.train()
        epoch_loss = 0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = yolo(images)

            loss = criterion(predictions, targets)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    torch.save(yolo.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

    plot_losses(train_losses)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv4 model")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training CSV file")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the images directory")
    parser.add_argument("--labels_dir", type=str, required=True, help="Path to the labels directory")
    parser.add_argument("--bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_anchors", type=int, default=3, help="Number of anchors per scale")
    parser.add_argument("--save_path", type=str, default="yolov4.pth", help="Path to save the trained model")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
    # python train.py --train_csv "/Users/armandbryan/Downloads/archive/train.csv" --images_dir "/Users/armandbryan/Downloads/archive/images" --labels_dir "/Users/armandbryan/Downloads/archive/labels" --save_path "/Users/armandbryan/Downloads/archive/yolo4.pth"





