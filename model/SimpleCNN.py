import torch
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation

class SimpleCNN(nn.Module):
    def __init__(self,num_classes,num_channels,window_len) -> None:
        super().__init__()
        
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
            Conv2dNormActivation(8,
                                 16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 norm_layer=nn.BatchNorm2d,
                                 activation_layer=nn.ReLU,
                                 inplace=True,
                                 bias=False,
                                 ),
            Conv2dNormActivation(16,
                                 32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 norm_layer=nn.BatchNorm2d,
                                 activation_layer=nn.ReLU,
                                 inplace=True,
                                 bias=False,
                                 ),
        )

        self.avgpool = nn.AvgPool2d((3, 3))

        self.classifier = nn.Linear(32 * (num_channels // 3) * (window_len // 3), num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
# if __name__ == '__main__':
#     model = SimpleCNN(num_classes=6, num_channels=561, window_len=64)
#     # a = torch.rand(5,1,64,45)
#     # b = model.forward(a)
#     # print(b.shape)
#     from torchsummary import summary
#     from torchstat import stat
#     summary(model.cuda(), (1,64,561))
#     stat(model.cpu(), (1, 64, 561))