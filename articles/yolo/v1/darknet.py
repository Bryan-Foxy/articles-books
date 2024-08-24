import torch
from articles.yolo.config import device
from articles.yolo.architecture import architecture_config

class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvolutionBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class YOLOv1(torch.nn.Module):
    def __init__(self, in_channels = 3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._build_convlayers(self.architecture)
        self.fc = self._build_fc(**kwargs)
    
    def _build_convlayers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [ConvolutionBlock(
                    in_channels, x[1], kernel_size = x[0], 
                    stride = x[2],
                    padding = x[3] 
                )]
                in_channels = x[1]
                
            elif type(x) == str:
                layers += [torch.nn.MaxPool2d(kernel_size = 2, stride = 2)]
            
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        ConvolutionBlock(in_channels, conv1[1], kernel_size = conv1[0],
                                         stride = conv1[2], padding = conv1[3])
                    ]

                    layers += [
                        ConvolutionBlock(conv1[1], conv2[1],
                                         kernel_size = conv2[0], padding = conv2[3])
                    ]

                    in_channels = conv2[1]
        
        return torch.nn.Sequential(*layers)
    
    def _build_fc(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024 * S * S, 496),
            torch.nn.Dropout(0.0),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(496, S * S * (C + B * 5)),
            )



    
    def forward(self, x):
        x = self.darknet(x)
        return self.fc(torch.flatten(x, start_dim = 1))

    
if __name__ == "__main__":
    device = device
    model = YOLOv1(split_size = 7, num_boxes = 2, num_classes = 10).to(device)
    # Store the architecture in a text file
    with open('darknet_architecture.txt', 'w') as f:
        print(model, file = f)

    sample_input = torch.randn((2, 3, 448, 448)).to(device)  # Batch size 1, 3 channels (RGB), 224x224 image
    output = model(sample_input)
    print(output.shape)  # Expected output shape: (1, 1000)



    