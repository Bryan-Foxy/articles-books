import torch
import argparse
from tqdm import tqdm
from dataset import PascalVOCDataset
from src.models.yolo import YOLO
from src.utils.config import get_device, get_num_classes

def train(args):
	device = get_device()
	print(f"Using device: {device}")
	train_dset = PascalVOCDataset(
		csv_path = args.train_csv,
		images_dir = args.images_dir,
		labels_dir = args.labels_dir
		)
	train_loader = torch.utils.data.DataLoader(
		train_dset,
		batch_size = args.bs,
		shuffle = True
		)

	global nc 
	nc = get_num_classes(args.train_csv)
	print(f"Number of classes: {nc}")
	yolo = YOLO(num_classes = nc, num_anchors = args.num_anchors).to(device)
	optimizer = torch.optim.Adam(yolo.parameters(), lr = args.lr)
	criterion = Loss(num_classes = nc).to(device)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

	# Training loop
    for epoch in range(args.epochs):
        yolo.train()
        epoch_loss = 0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (images, targets) in enumerate(loop):
            images, targets = images.to(device), {k: v.to(device) for k, v in targets.items()}

            # Forward pass
            predictions = yolo(images)

            # Compute loss
            loss = criterion(predictions, targets)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            loop.set_postfix(loss=loss.item())

        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv4 model")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training CSV file")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the images directory")
    parser.add_argument("--labels_dir", type=str, required=True, help="Path to the labels directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=nc, help="Number of object classes")
    parser.add_argument("--num_anchors", type=int, default=3, help="Number of anchors per scale")
    parser.add_argument("--save_path", type=str, default="yolov4.pth", help="Path to save the trained model")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()





