import torch
import torch.nn as nn



class VGGNet(nn.Module):
    def __init__(self, input_channels, output_classes, architecture="VGG16"):
        super(VGGNet, self).__init__()

        VGG = {
            "VGG16": [64, 64, "p", 128, 128, "p", 256, 256, 256, "p", 512, 512, 512, "p", 512, 512, 512, "p"],
            "VGG19": [64, 64, "p", 128, 128, "p", 256, 256, 256, "p", 512, 512, 512, "p", 512, 512, 512, "p", 512, 512, 512, "p"]
        }[architecture]

        self.input_channels = input_channels
        self.output_classes = output_classes

        self.model = self._build_architecture(VGG)
        

    def forward(self, x):
        return self.model(x)

    def _build_architecture(self, architecture):
        layers = []
        count_max_pool_layers = 0
        in_channels = self.input_channels

        # adding convolutional layers
        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                    ]
                
                in_channels = x
            else:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
                count_max_pool_layers += 1


        conv_output_channels = (224 // (2**count_max_pool_layers))**2

        # adding fully connected layers
        layers += [
            nn.Flatten(),
            nn.Linear(512*conv_output_channels, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.output_classes)
        ]

        

        return nn.Sequential(*layers)