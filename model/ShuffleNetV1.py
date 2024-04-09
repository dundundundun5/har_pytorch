from typing import Any, List

import torch
from torch import Tensor
from torch import nn

#   model = ShuffleNetV1([4, 8, 4], [16, 192, 384, 768], 8, **kwargs)
class ShuffleNetV1(nn.Module):

    def __init__(
            self,
            num_classes,
            num_channels,
            window_len,
            groups: int = 8,
    ) -> None:
        super(ShuffleNetV1, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )

        features = nn.Sequential(
            ShuffleNetV1Unit(
                        8,
                        16,
                        1,
                        groups,
                        True,
                    ),
            ShuffleNetV1Unit(
                        16,
                        32,
                        1,
                        groups,
                        False,
                    )
        )
        self.features = nn.Sequential(*features)

        self.globalpool = nn.AvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Linear(32 * (num_channels // 3) * (window_len // 3), num_classes),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        out = self.first_conv(x)
        out = self.features(out)
        out = self.globalpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    


class ShuffleNetV1Unit(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            groups: int,
            first_groups: bool = False,
    ) -> None:
        super(ShuffleNetV1Unit, self).__init__()
        self.stride = stride
        self.groups = groups
        self.first_groups = first_groups
        hidden_channels = out_channels // 4

        self.branch_main_1 = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_channels, (1, 1), (1, 1), (0, 0), groups=1 if first_groups else groups,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            # dw
            nn.Conv2d(hidden_channels, hidden_channels, (3, 3), (stride, stride), (1, 1), groups=hidden_channels,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.branch_main_2 = nn.Sequential(
            # pw-linear
            nn.Conv2d(hidden_channels, out_channels, (1, 1), (1, 1), (0, 0), groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(True)

    def channel_shuffle(self, x):
        batch_size, channels, height, width = x.data.size()
        assert channels % self.groups == 0
        group_channels = channels // self.groups

        out = x.reshape(batch_size, group_channels, self.groups, height, width)
        out = out.permute(0, 2, 1, 3, 4)
        out = out.reshape(batch_size, channels, height, width)

        return out

    def forward(self, x: Tensor) -> Tensor:

        out = self.branch_main_1(x)
        out = self.channel_shuffle(out)
        out = self.branch_main_2(out)
        out = self.relu(out)
        return out


# if __name__ == '__main__':
#     model = ShuffleNetV1(num_channels=561, num_classes=6, window_len=64, groups=2)

# #   a = torch.rand(16, 1, 64, 45)
# #   b = model.forward(a)
# #   print(b.shape)
#     from torchsummary import summary
#     from torchstat import stat
#     summary(model.cuda(), (1,64,561))
#     stat(model.cpu(), (1, 64, 561))