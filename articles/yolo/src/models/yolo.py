import torch
from models.neck import PANet
from models.head import YOLOHead
from models.backbone import CSPDarknet

class YOLO(torch.nn.Module):
	def __init__(self, num_classes, num_anchors):
		super(YOLO, self).__init__()
		self.backbone = CSPDarknet()
		self.neck = PANet([64, 128, 256])
		self.head = YOLOHead(num_classes, num_anchors, [64, 128, 256])

	def forward(self, x):
		features = self.backbone(x)
		neck_outputs = self.neck(features)
		predictions = self.head(neck_outputs)
		return predictions