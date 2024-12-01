import torch

class Loss(torch.nn.Module):
    """
    Loss function for YOLOv4.
    Combines:
        - Localization Loss (Bounding Box Regression)
        - Objectness Loss (Binary Cross-Entropy)
        - Classification Loss (Cross-Entropy for classes)
    """
    def __init__(self, num_classes, lambda_coord=5.0, lambda_noobj=0.5):
        """
        Args:
            num_classes (int): Number of classes.
            lambda_coord (float): Weight for bounding box regression loss.
            lambda_noobj (float): Weight for no-objectness loss.
        """
        super(Loss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        # Loss functions
        self.mse_loss = torch.nn.MSELoss(reduction="sum")  # For bbox regression
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")  # For objectness
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="sum")  # For classification

    def forward(self, predictions, targets):
        total_loss = 0.0

        for scale_idx, pred in enumerate(predictions):
            batch_loss = 0.0
            for batch_idx, batch_pred in enumerate(pred):
                # Get targets for this specific image
                target_boxes = targets[batch_idx]['boxes']
                target_labels = targets[batch_idx]['labels']

                # If no objects in this image, skip
                if len(target_boxes) == 0:
                    continue

                # Ensure predictions are in the right shape
                batch_pred = batch_pred.reshape(-1, self.num_classes + 5)

                # Compute losses
                box_loss = self.lambda_coord * self.mse_loss(
                    batch_pred[:len(target_boxes), :4], 
                    target_boxes
                )

                # Objectness loss (placeholder - you might need more sophisticated handling)
                objectness_target = torch.ones(len(target_boxes), device=batch_pred.device)
                object_loss = self.bce_loss(
                    batch_pred[:len(target_boxes), 4], 
                    objectness_target
                )

                # Classification loss
                class_loss = self.ce_loss(
                    batch_pred[:len(target_boxes), 5:], 
                    target_labels
                )

                batch_loss += box_loss + object_loss + class_loss

            # Average loss over the batch
            if batch_loss > 0:
                total_loss += batch_loss / len(predictions[scale_idx])

        return total_loss