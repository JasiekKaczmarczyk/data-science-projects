import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models


class VGG_NST(nn.Module):
    def __init__(self, feature_layers):
        super(VGG_NST, self).__init__()

        self.feature_layers = feature_layers

        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_idx, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_idx) in self.feature_layers:
                features.append(x)

        return features
