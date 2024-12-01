# Author: FOZAME ENDEZOUMOU Armand Bryan
# Date: 23/11/2024
# Date: 20:38:43

# CSPDarknet (Cross-Stage Partial Darknet) is the backbone network used in YOLOv4 for feature extraction. 
# It builds on Darknet-53 by introducing cross-stage partial connections to reduce computation, memory usage, and redundancy in gradient flow, while improving accuracy.

import torch

class ConvBlock(torch.nn.Module):
    """Convolution block conv -> BN -> activation function"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.Mish()  # I choose Mish here. But the classic is LeakyReLU

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x


class ResidualBlock(torch.nn.Module):
    """Residual Block conv -> conv -> result(conv2) + input"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        final_x = residual + x
        return final_x


class CSPBlock(torch.nn.Module):
    """
    Cross Stage Partial Block. It's used to be more efficient, less computational.
    Split the input into two parts, one will be used on residual block, and the output 
    will be concatenated with the second part of the input split.
    """
    def __init__(self, in_channels, out_channels, num_residual_blocks):
        super(CSPBlock, self).__init__()
        self.split_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(out_channels // 2, out_channels // 4) for _ in range(num_residual_blocks)]
        )

        self.conv_concatenated = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        split = self.split_conv(x)
        residual_output = self.residual_blocks(split)
        x = torch.cat([split, residual_output], dim=1)
        x = self.conv_concatenated(x)
        return x


class CSPDarknet(torch.nn.Module):
    """CSPDarknet"""
    def __init__(self):
        super(CSPDarknet, self).__init__()
        self.stem = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        
        # Downsampling and CSP layers
        self.layer1 = torch.nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            CSPBlock(64, 64, num_residual_blocks=1),
        )  # Output: P3 (High resolution)

        self.layer2 = torch.nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            CSPBlock(128, 128, num_residual_blocks=2),
        )  # Output: P4 (Intermediate resolution)

        self.layer3 = torch.nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            CSPBlock(256, 256, num_residual_blocks=8),
        )  # Output: P5 (Low resolution)

    def forward(self, x):
        x = self.stem(x)
        P3 = self.layer1(x)  
        P4 = self.layer2(P3) 
        P5 = self.layer3(P4)
        return P3, P4, P5