from albumentations import (
    Compose, Resize, HorizontalFlip, VerticalFlip, ShiftScaleRotate,
    RandomBrightnessContrast, RandomGamma, HueSaturationValue, CLAHE,
    GaussianBlur, MotionBlur, CoarseDropout, Normalize
)
from albumentations.pytorch import ToTensorV2

def get_yolo_augmentation(image_size=416):
    """
    Return an augmentation pipeline tailored for YOLO.
    """
    transform = Compose(
        [
            # Resize the image and bounding boxes to fit YOLO input
            Resize(height=image_size, width=image_size),
            
            # Flipping for spatial variation
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.2),
            
            # Random rotations and scaling for better generalization
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            
            # Color augmentations to handle different lighting conditions
            RandomBrightnessContrast(p=0.3),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            RandomGamma(gamma_limit=(80, 120), p=0.3),
            CLAHE(clip_limit=2.0, p=0.2),
            
            # Adding blur to simulate motion or camera focus issues
            GaussianBlur(blur_limit=(3, 5), p=0.2),
            MotionBlur(blur_limit=3, p=0.2),
            
            # Dropout to randomly remove parts of the image for robustness
            CoarseDropout(
                max_holes=8, max_height=0.1, max_width=0.1,
                min_holes=1, min_height=0.05, min_width=0.05, fill_value=0, p=0.2
            ),
            
            # Normalize and convert to tensor
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ],
        bbox_params={
            "format": "yolo",  # YOLO bounding box format (x_center, y_center, width, height)
            "label_fields": ["class_labels"],  # Ensure bounding boxes and labels are synced
        }
    )
    return transform