import torch
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
# 仅在PAMAP2上的源码

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(1, 0), dilation=1, groups=1, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) 
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = (5, 1)
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(2, 0))

    def forward(self, x):

        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()

        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.hw = AttentionGate()
        
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        
        self.w1.data.fill_(1/3)
        self.w2.data.fill_(1/3)
        self.w3.data.fill_(1/3)

            

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out = self.hw(x)
        x_out = self.w1 * x_out + self.w2 * x_out11 + self.w3 * x_out21

       
        return x_out
    
class TripleAttentionCNN(nn.Module):

    def __init__(self, window_len, num_channels,num_classes, out_channels=64):
        super(TripleAttentionCNN, self).__init__()
        conv1 = nn.Conv2d(1, out_channels, (3, 1), (3, 1), padding=(1, 0))
        att1 = TripletAttention()
        window_len = (window_len - 3 + 2) // 3 + 1
        
        conv2 = nn.Conv2d(out_channels, out_channels*2, (3, 1), (3, 1), padding=(1, 0))
        att2 = TripletAttention()
        window_len = (window_len - 3 + 2) // 3 + 1
        
        conv3 = nn.Conv2d(out_channels*2, out_channels*4, (3, 1), (3, 1), padding=(1, 0))
        att3 = TripletAttention()
        window_len = (window_len - 3 + 2) // 3 + 1
        
        self.conv_module = nn.Sequential(
            conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            att1,

            conv2,
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True),
            att2,

            conv3,
            nn.BatchNorm2d(out_channels*4),
            nn.ReLU(inplace=True),
            att3
        )
        # 计算输入特征数
        # 参数化输出类别
        self.classifier = nn.Sequential(
            nn.Linear(in_features=window_len*(out_channels*4)*num_channels, out_features=num_classes)
        )                            

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_module(x) # [B,1,L,C] -> [B,256,,C]
        x = x.reshape(x.shape[0], -1)
        # print(x.shape, 'x')
        x = self.classifier(x)
        return x
class TripleAttentionResnet(nn.Module):
    def __init__(self, window_len, num_channels,num_classes, out_channels=64):
        super(TripleAttentionResnet, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.att1 = TripletAttention()

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
        )
        
        
        window_len = (window_len - 6 + 2) // 3 + 1
        window_len = (window_len - 3 + 2) // 1 + 1
        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(True)
        )
        self.att2 = TripletAttention()

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels*2),
        )
        window_len = (window_len - 6 + 2) // 3 + 1
        window_len = (window_len - 3 + 2) // 1 + 1
        
        
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*4, kernel_size=(3, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels*4),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*4, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels*4),
            nn.ReLU(True)
        )
        self.att3 = TripletAttention()

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*4, kernel_size=(3, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels*4),
        )
        window_len = (window_len - 3 + 2) // 3 + 1
        window_len = (window_len - 3 + 2) // 1 + 1
        
        
        self.fc = nn.Sequential(
            nn.Linear(window_len*(out_channels*4)*num_channels, num_classes)
        )

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        out1 = self.Block1(x)
        out1 = self.att1(out1)
        y1 = self.shortcut1(x)
        out = y1 + out1

        out2 = self.Block2(out)
        out2 = self.att2(out2)
        y2 = self.shortcut2(out)
        out = y2 + out2

        out3 = self.Block3(out)
        out3 = self.att3(out3)
        y3 = self.shortcut3(out)
        out = y3 + out3

        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda()
        return out    
if __name__ == "__main__":

    l,c, cl = 128,12, 14
    model = TripleAttentionCNN(l, c, cl)
    a = torch.rand(33,1 ,l, c)
    
    print(model(a))
    pass