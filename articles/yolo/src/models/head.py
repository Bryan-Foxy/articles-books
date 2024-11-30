import torch

class YOLOHead(torch.nn.Module):
	def __init__(self, num_classes, num_anchors, in_channels_list):
		super(YOLOHead, self).__init__()
		self.num_classes = num_classes
		self.num_anchors = num_anchors
		self.prediction_layers = torch.nn.ModuleList([
			torch.nn.Conv2d(in_channels, num_anchors * (num_classes + 5), kernel_size = 1)
			for in_channels in in_channels_list
			])

	def forward(self, features):
		predictions = []
		for feature_map, prediction_layer in zip(features, self.prediction_layers):
			prediction = prediction_layer(feature_map)
			bs, _, grid_size, _ = prediction.shape
			prediction = prediction.view(
				bs,
				self.num_anchors,
				self.num_classes + 5,
				grid_size,
				grid_size).permute(0, 3, 4, 1, 2) # Shape: (Batch, Grid Size, Grid Size, Anchors, Classes + 5)
			predictions.append(prediction)

		return predictions