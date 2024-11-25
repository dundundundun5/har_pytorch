# TODO GRU-INC复现
from torch import nn
import torch
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
class InceptionV2(nn.Module):
    def __init__(self, in_channels, filter=8) -> None:
        super(InceptionV2, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,  out_channels=filter, kernel_size=(1,1)),
            nn.BatchNorm2d(filter),
            nn.ReLU(True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,  out_channels=filter, kernel_size=(1,1)),
            nn.BatchNorm2d(filter),
            nn.ReLU(True),
            nn.Conv2d(in_channels=filter,  out_channels=filter, kernel_size=(1,3), padding=(0, 1)),
            nn.BatchNorm2d(filter),
            nn.ReLU(True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,  out_channels=filter, kernel_size=(1,1)),
            nn.BatchNorm2d(filter),
            nn.ReLU(True),
            nn.Conv2d(in_channels=filter,  out_channels=filter, kernel_size=(1,3), padding=(0, 1)),
            nn.BatchNorm2d(filter),
            nn.ReLU(True),
            nn.Conv2d(in_channels=filter,  out_channels=filter, kernel_size=(1,3), padding=(0, 1)),
            nn.BatchNorm2d(filter),
            nn.ReLU(True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,  out_channels=filter, kernel_size=(1,1)),
            nn.BatchNorm2d(filter),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1,3), stride=1,padding=(0, 1))
        )
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.concat([b1,b2,b3,b4], dim=1)
        return out
class GRUINC(nn.Module):
    
    def __init__(self, window_len, num_channels, num_classes, filter=16) -> None:
        super(GRUINC, self).__init__()
        self.num_channels = num_channels
        self.gru1 = nn.GRU(window_len, hidden_size=256, batch_first=True)
        self.att1 = nn.Sequential(
            nn.Linear(in_features=256*num_channels, out_features=256*num_channels),
            nn.Softmax()
        )
        self.gru2 = nn.GRU(input_size=256, hidden_size=244, batch_first=True)
        self.inception1 = InceptionV2(in_channels=1, filter=filter)
        self.inception2 = InceptionV2(in_channels=filter*4, filter=filter*4)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,1), stride=1)
        self.dropout = nn.Dropout()
        self.cbam = CBAM(filter*16)
        self.inception3 = InceptionV2(filter*16, filter=filter)
        self.avgpool =  nn.AvgPool2d(kernel_size=(1,1), stride=1)
        self.dropout2 = nn.Dropout()
        self.fc = nn.Linear(in_features=filter*4*244*num_channels, out_features=num_classes)
    def forward(self, x):
        # x [batch, window, channel] -> [batch, channel, window]
        x = x.permute(0, 2, 1)
        x, _ = self.gru1(x, None)
        
        x = x.reshape(x.shape[0], -1)
        att_gru1 = self.att1(x)
        x = x * att_gru1
        x = x.reshape(x.shape[0], self.num_channels, -1)
        x, _ = self.gru2(x, None)
        x = x.permute(0, 2, 1) # [batch, gru2_hidden, channel]
        x = x.unsqueeze(1)
        
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool(x)
        
        x = x + self.dropout(self.cbam(x))
        x = self.inception3(x)
        
        x = self.avgpool(x)
        x = self.dropout2(x)
        x = x.reshape(x.shape[0], -1)
        out = self.fc(x)
        return out
if __name__  == '__main__':
    
    
    
    a = torch.rand(16,64, 113)
    
    model = GRUINC(64, 113, 6)
    
    print(model(a))