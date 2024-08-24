import torch
from ...functions import iou

class LossFunctionYOLO(torch.nn.Module):
    def __init__(self, S, B, C, lambda_noobject = 0.5, lambda_coord = 5):
        super(LossFunctionYOLO, self).__init__()
        self.mse = torch.nn.MSELoss(reduction = "sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobject = lambda_noobject
        self.lambda_coord = lambda_coord
    
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_box1 = iou(predictions[..., 21:25], target[..., 21:25])
        iou_box2 = iou(predictions[..., 26:30], target[..., 21:25])
        all_iou = torch.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim = 0)
        best_iou, best_box = torch.max(all_iou, dim = 0) 
        box_ = target[..., 20].unsqueeze(3)

        # Loss
        box_pred = box_ * ((
            best_box * predictions[..., 26:30] * (1 - best_box) * predictions[..., 21:25]
        ))
        box_target = box_ * target[..., 21:25]
        box_pred[..., 2:4] = torch.sign(box_pred[...,2:4]) * torch.sqrt(torch.abs(box_pred[..., 2:4] + 1e-6)) 
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])
        box_loss = self.mse(torch.flatten(box_pred, end_dim = -2), torch.flatten(box_target, end_dim = -2))

        # Object Loss
        box_pred_obj = (best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21])
        obj_loss = self.mse(torch.flatten(box_ *box_pred_obj), torch.flatten(box_ * target[..., 20:21]))

        # No Object Loss
        no_obj_loss = self.mse(torch.flatten((1 - box_) * predictions[..., 20:21], start_dim = 1), torch.flatten((1 - box_) * target[..., 20:21], start_dim = 1))
        no_obj_loss += self.mse(torch.flatten((1 - box_) * predictions[..., 25:26], start_dim = 1), torch.flatten((1 - box_) * target[..., 20:21], start_dim = 1))

        # Class Loss
        class_loss = self.mse(torch.flatten(box_ *predictions[..., :20], end_dim = -2), torch.flatten(box_ * target[..., :20], start_dim = -2))

        final_loss = ((self.lambda_coord * box_loss) + obj_loss + (self.lambda_noobject * no_obj_loss) + class_loss)
        return final_loss








