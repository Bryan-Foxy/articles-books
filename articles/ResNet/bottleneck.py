import torch
import torch.nn.functional as F

class Bottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, factor = 4, stride = 1):
        super(Bottleneck, self).__init__()
        self.factor = factor 
        self.conv1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride, bias = False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(in_channels = out_channels, out_channels = out_channels * self.factor, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * self.factor)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.factor:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels * self.factor, kernel_size = 1, stride = stride, bias = False),
                torch.nn.BatchNorm2d(out_channels * self.factor)
            )
    

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    



if __name__ == "__main__":
    res_block = Bottleneck(64,64, stride = 1)
    print(res_block)
    x = torch.randn(1,64,60,60)
    test = res_block(x)
    print(test)
