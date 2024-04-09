from typing import Callable, Any, Optional
import torch
from torch import Tensor
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation
class MobileNetV1(nn.Module):

    def __init__(
            self,
            num_classes,
            num_channels,
            window_len
    ) -> None:
        super(MobileNetV1, self).__init__()
        self.features = nn.Sequential(
            Conv2dNormActivation(1,
                                 8,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 norm_layer=nn.BatchNorm2d,
                                 activation_layer=nn.ReLU,
                                 inplace=True,
                                 bias=False,
                                 ),

            DepthWiseSeparableConv2d(8, 16, 1),
            DepthWiseSeparableConv2d(16, 32, 1),
        )

        self.avgpool = nn.AvgPool2d((3, 3))

        self.classifier = nn.Linear(32 * (num_channels // 3) * (window_len // 3), num_classes)


    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class DepthWiseSeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DepthWiseSeparableConv2d, self).__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            Conv2dNormActivation(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 padding=1,
                                 groups=in_channels,
                                 norm_layer=norm_layer,
                                 activation_layer=nn.ReLU,
                                 inplace=True,
                                 bias=False,
                                 ),
            Conv2dNormActivation(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 norm_layer=norm_layer,
                                 activation_layer=nn.ReLU,
                                 inplace=True,
                                 bias=False,
                                 ),

        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        return out
# if __name__ == '__main__':
#     model = MobileNetV1(num_classes=6, num_channels=561, window_len=64)
#     from torchsummary import summary
#     from torchstat import stat
#     summary(model.cuda(), (1,64,561))
#     stat(model.cpu(), (1, 64, 561))