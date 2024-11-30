from neck import PANet
from head import YOLOHead
from backbone import CSPDarknet

class YOLO(torch.nn.Module):
	def __init__(self, num_classes, num_anchors):
		super(YOLO, self).__init__()
		self.backbone = CSPDarknet()
		self.neck = PANet([256, 512, 1024])
		self.head = YOLOHead(num_classes, num_anchors, [256, 512, 1024])

	def forward(self, x):
		features = self.backbone(x)
		neck_outputs = self.neck(features)
		predictions = self.head(neck_outputs)
		predictions