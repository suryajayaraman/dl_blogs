import torch
import torch.nn as nn


# Darknet 53 backbone
BACKBONE_CONFIG = [
    (3, 32, 1, 1),
    (3, 64, 2, 1),
    [(1, 32, 1, 0), (3, 64, 1, 1), 1],
    (3, 128, 2, 1),
    [(1, 64, 1, 0), (3, 128, 1, 1), 2],
    (3, 256, 2, 1),
    [(1, 128, 1, 0), (3, 256, 1, 1), 8],
    (3, 512, 2, 1),
    [(1, 256, 1, 0), (3, 512, 1, 1), 8],
    (3, 1024, 2, 1),
    [(1, 512, 1, 0), (3, 1024, 1, 1), 4]    
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, **kwargs) -> None:
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False,**kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class yoloV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs) -> None:
        super(yoloV1, self).__init__()
        self.in_channels = in_channels
        self.darknet = self.createBackboneLayers(BACKBONE_CONFIG)
        self.fcLayers = self.createFcLayers(**kwargs)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.darknet(x)
        return self.fcLayers(torch.flatten(x, start_dim=1))


    def createFcLayers(self, splitSize, numBoxes, numClasses) -> nn.Module:
        S, B, C = splitSize, numBoxes, numClasses
        return nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(1024 * S * S, 512),
                    nn.LeakyReLU(0.1),
                    nn.Linear(512, S * S * (B*5 + C))
                )

    def createBackboneLayers(self, archConfig:list) -> nn.Module:
        layers = []
        in_channels = self.in_channels

        for moduleData in archConfig:
            if(type(moduleData) == tuple):
                layers += [CNNBlock(in_channels, moduleData[1],
                                    kernel_size = moduleData[0],
                                    stride = moduleData[2],
                                    padding = moduleData[3])]
                in_channels = moduleData[1]

            elif(type(moduleData) == str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
            elif(type(moduleData) == list):
                conv1Data = moduleData[0]
                conv2Data = moduleData[1]
                numRepeats = moduleData[2]
                
                for _ in range(numRepeats):
                    layers += [CNNBlock(in_channels, conv1Data[1],
                                    kernel_size = conv1Data[0],
                                    stride = conv1Data[2],
                                    padding = conv1Data[3])]

                    layers += [CNNBlock(conv1Data[1], conv2Data[1],
                                    kernel_size = conv2Data[0],
                                    stride = conv2Data[2],
                                    padding = conv2Data[3])]
                    
                    in_channels = conv2Data[1]

            else:
                pass

        return nn.Sequential(*layers)

