import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class FeaturizerModel(nn.Module):
    def __init__(self):
        super(FeaturizerModel, self).__init__()
        self.encoder = nn.Sequential(# in- (BS,3,128, 128)

            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=2,
                      padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=1,
                      padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=2,
                      padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), #13

            nn.Conv2d(in_channels=256, #14
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1),
            nn.ReLU(True), # 15
            nn.Conv2d(in_channels=512, #16
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=0),
            nn.ReLU(True),            # 17
            nn.MaxPool2d(2, stride=2) # 18
        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(in_channels=512, #20
                               out_channels=512,
                               kernel_size=(3,3),
                               stride=1,
                              padding=1),
            nn.ReLU(True), #22
            nn.ConvTranspose2d(in_channels=512, #23
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=0),

            nn.ConvTranspose2d(in_channels=256, #24
                               out_channels=128,
                               kernel_size=(3,3),
                               stride=2,
                               padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=(3,3),
                               stride=2,
                               padding=1),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=64,
                               kernel_size=(3,3),
                               stride=2,
                               padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=(3,3),
                               stride=2,
                               padding=1),

            nn.ConvTranspose2d(in_channels=32,
                               out_channels=32,
                               kernel_size=(3,3),
                               stride=2,
                               padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32,
                               out_channels=3,
                               kernel_size=(4,4),
                               stride=2,
                               padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x