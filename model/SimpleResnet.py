import torch
from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3)),
            nn.BatchNorm2d(out_channels)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.block(x) + self.residual(x)
        return out


class SimpleResnet(nn.Module):
    def __init__(self, window_len, num_channels, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_len = window_len
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.resnet = nn.Sequential(
            ResnetBlock(in_channels=1, out_channels=16),
            ResnetBlock(in_channels=16, out_channels=32),
            ResnetBlock(in_channels=32, out_channels=64)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=window_len*num_channels*64, out_features=self.num_classes),
            nn.Linear(in_features=self.num_classes, out_features=self.num_classes),

        )
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        # out = nn.functional.softmax(out, dim=1)
        return out


if __name__ == '__main__':
    model = SimpleResnet(window_len=32, num_channels=7, num_classes=18)
    x = torch.rand(9, 1, 32, 7)
    b = model.forward(x)
    print(b)
