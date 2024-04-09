import torch
import torch.nn as nn
import torch.nn.functional as F
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // out_channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // out_channels, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class DualAttentionCNN(nn.Module):
    def __init__(self, window_len, num_channels, num_classes, out_channels=64):
        super(DualAttentionCNN, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        window_len = (window_len-3) // 2 + 1
        self.ca1 = ChannelAttention(out_channels, out_channels)
        self.sa1 = SpatialAttention()

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(True)
        )
        window_len = (window_len-3) // 2 + 1
        self.ca2 = ChannelAttention(out_channels * 2,out_channels)
        self.sa2 = SpatialAttention()

        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 4, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(True)
        )
        window_len = (window_len-3) // 2 + 1
        self.ca3 = ChannelAttention(out_channels * 4,out_channels)
        self.sa3 = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(window_len*(out_channels * 4)*num_channels, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        h1 = self.Block1(x)
        h1 = self.ca1(h1) * h1
        h1=  self.sa1(h1) * h1

        h2 = self.Block2(h1)
        h2 = self.ca2(h2) * h2
        h2 = self.sa2(h2) * h2
        
        h3 = self.Block3(h2)
        h3 = self.ca3(h3) * h3
        h3 = self.sa3(h3) * h3

        x = h3.view(h3.size(0), -1)

        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()

        return x
class DualAttentionResnet(nn.Module):
    def __init__(self, window_len, num_channels, num_classes, out_channels=128):
        super(DualAttentionResnet, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.ca1 = ChannelAttention(out_channels, out_channels)
        self.sa1 = SpatialAttention()
        window_len = (window_len - 6 + 2) // 2 + 1
        window_len = (window_len - 3 + 2) // 1 + 1
        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(True)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels*2),
        )
        self.ca2 = ChannelAttention(out_channels*2, out_channels)
        self.sa2 = SpatialAttention()
        window_len = (window_len - 6 + 2) // 3 + 1
        window_len = (window_len - 3 + 2) // 1 + 1
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*2, out_channels=int(out_channels*1.5), kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(int(out_channels*1.5)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=int(out_channels*1.5), out_channels=int(out_channels*1.5), kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(int(out_channels*1.5)),
            nn.ReLU(True)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*2, out_channels=int(out_channels*1.5), kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(int(out_channels*1.5)),
        )
        self.ca3 = ChannelAttention(int(out_channels*1.5), out_channels)
        self.sa3 = SpatialAttention()
        window_len = (window_len - 6 + 2) // 2 + 1
        window_len = (window_len - 3 + 2) // 1 + 1
        self.fc = nn.Sequential(
            nn.Linear(window_len*(int(out_channels * 1.5))*num_channels, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)
        h1 = self.Block1(x)
        # print(h1.shape)
        r = self.shortcut1(x)
        # print(r.shape)
        h1 = self.ca1(h1) * h1
        
        h1=  self.sa1(h1) * h1
        
        h1 = h1 + r
        # print(h1.shape)
        h2 = self.Block2(h1)
        # print(h2.shape)
        r = self.shortcut2(h1)
        # print(r.shape)
        h2 = self.ca2(h2) * h2
        h2 = self.sa2(h2) * h2
        h2 = h2 + r
        # print(h2.shape)
        h3 = self.Block3(h2)
        # print(h3.shape)
        r = self.shortcut3(h2)
        # print(r.shape)
        h3 = self.ca3(h3) * h3
        h3 = self.sa3(h3) * h3
        h3 = h3 + r
        x = h3.view(h3.size(0), -1)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        return x
if __name__ == "__main__":
    x = torch.rand(32, 151, 3).cuda()
    model = DualAttentionResnet(151, 3, 7 ,16).cuda()
    model2 = DualAttentionCNN(151, 3, 7 ,16).cuda()
    b = model(x)
    print(b.shape)