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

        for scale_idx, (pred, target) in enumerate(zip(predictions, targets)):
            pred_boxes = pred[..., :4]  # (x, y, w, h)
            pred_objectness = pred[..., 4]  # Objectness score
            pred_classes = pred[..., 5:]  # Class probabilities

            
            target_boxes = target["boxes"] 
            target_objectness = target["objectness"]  
            target_classes = target["class_ids"] 

            obj_mask = target_objectness > 0
            box_loss = self.lambda_coord * self.mse_loss(
                pred_boxes[obj_mask], target_boxes[obj_mask]
            )

            object_loss = self.bce_loss(pred_objectness[obj_mask], target_objectness[obj_mask])
            no_object_loss = self.lambda_noobj * self.bce_loss(
                pred_objectness[~obj_mask], target_objectness[~obj_mask]
            )

            class_loss = self.ce_loss(
                pred_classes[obj_mask], target_classes[obj_mask]
            )

            scale_loss = box_loss + object_loss + no_object_loss + class_loss
            total_loss += scale_loss

        return total_loss