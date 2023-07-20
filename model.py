"""Autoencoder model for image compression using AlexNet as encoder"""
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = alexnet(weights=AlexNet_Weights.DEFAULT).features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2),  # upsample
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(192, 192, kernel_size=3, stride=2),  # upsample
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # upsample
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=11, stride=4, padding=2, output_padding=1),  # upsample
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),

            nn.Conv2d(3, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
