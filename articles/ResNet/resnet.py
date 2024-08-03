from residual_block import ResidualBlock
import torch.nn.functional as F
import torch

class ResNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.initial_conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.initial_bn = torch.nn.BatchNorm2d(32)
        self.layer1 = ResidualBlock(in_channels=32, out_channels=32, stride=1)
        self.layer2 = ResidualBlock(in_channels=32, out_channels=64, stride=2)
        self.layer3 = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        self.layer4 = ResidualBlock(in_channels=128, out_channels=256, stride=2)
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    res_net = ResNet()
    print(res_net)
    x = torch.randn(1, 3, 32, 32)
    test = res_net(x)
    print(test)