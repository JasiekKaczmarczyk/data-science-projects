import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_channels, discriminator_features):
        super(Discriminator, self).__init__()

        self.image_channels = image_channels
        self.discriminator_features = discriminator_features

        self.discriminator = self._build_architecture()

    def _build_architecture(self):
        return nn.Sequential(
            nn.Conv2d(self.image_channels, self.discriminator_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._add_convolution_block(self.discriminator_features, self.discriminator_features*2, kernel_size=4, stride=2, padding=1),
            self._add_convolution_block(self.discriminator_features*2, self.discriminator_features*4, kernel_size=4, stride=2, padding=1),
            self._add_convolution_block(self.discriminator_features*4, self.discriminator_features*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(self.discriminator_features*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def _add_convolution_block(self, input_channels, output_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.discriminator(x)


class Generator(nn.Module):
    def __init__(self, z_dim, image_channels, generator_features):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.image_channels = image_channels
        self.generator_features = generator_features

        self.generator = self._build_architecture()

    def _build_architecture(self):
        return nn.Sequential(
            self._add_transpose_convolution_block(self.z_dim, self.generator_features*16, kernel_size=4, stride=1, padding=0),
            self._add_transpose_convolution_block(self.generator_features*16, self.generator_features*8, kernel_size=4, stride=2, padding=1),
            self._add_transpose_convolution_block(self.generator_features*8, self.generator_features*4, kernel_size=4, stride=2, padding=1),
            self._add_transpose_convolution_block(self.generator_features*4, self.generator_features*2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(self.generator_features*2, self.image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _add_transpose_convolution_block(self, input_channels, output_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.generator(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

