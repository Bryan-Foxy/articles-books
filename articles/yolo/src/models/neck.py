# PANet model

import torch 

class ConvBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
		super(ConvBlock, self).__init__()
		self.conv = torch.nn.Conv2d(in_channels, 
			out_channels,
			kernel_size = kernel_size,
			stride = stride, 
			padding = padding)
		self.bn = torch.nn.BatchNorm2d(out_channels)
		self.act = torch.nn.LeakyReLU(0.2)

	def forward(self, x):
		return self.act(self.bn(self.conv(x)))

class PANet(torch.nn.Module):
	def __init__(self, in_channels_list):
		super(PANet, self).__init__()
		self.top_down_conv1 = ConvBlock(in_channels_list[2], in_channels_list[1])
		self.top_down_conv2 = ConvBlock(in_channels_list[1], in_channels_list[0])
		self.bottom_up_conv1 = ConvBlock(in_channels_list[0], in_channels_list[1])
		self.bottom_up_conv2 = ConvBlock(in_channels_list[1], in_channels_list[2])
		self.final_conv1 = ConvBlock(in_channels_list[0], in_channels_list[0])
		self.final_conv2 = ConvBlock(in_channels_list[1], in_channels_list[1])
		self.final_conv3 = ConvBlock(in_channels_list[2], in_channels_list[2])

	def forward(self, features):
		P3, P4, P5 = features
		P4_td = self.top_down_conv1(P5)
		P4_td = torch.nn.functional.interpolate(P4_td, scale_factor = 2, mode = "nearest") + P4
		P3_td = self.top_down_conv2(P4_td)
		P3_td = torch.nn.functional.interpolate(P3_td, scale_factor = 2, mode = "nearest") + P3
		P4_bu = self.bottom_up_conv1(P3_td)
		P4_bu = torch.nn.functional.max_pool2d(P4_bu, kernel_size = 2, stride = 2) + P4_td
		P5_bu = self.bottom_up_conv2(P4_bu)
		P5_bu = torch.nn.functional.max_pool2d(P5_bu, kernel_size = 2, stride = 2) + P5
		P3_out = self.final_conv1(P3_td)
		P4_out = self.final_conv2(P4_bu)
		P5_out = self.final_conv3(P5_bu)

		return [P3_out, P4_out, P5_out]

