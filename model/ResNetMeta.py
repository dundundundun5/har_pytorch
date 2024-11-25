from model.meta import nn
# 仅在PAMAP2上的源码

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(1, 0), dilation=1, groups=1, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv_module = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self, x, meta_loss=None):
        out = self.conv_module(x, meta_loss)
        return out




class ResnetMeta(nn.Module):
    def __init__(self, window_len, num_channels,num_classes, out_channels=64):
        super(ResnetMeta, self).__init__()
        self.feature = None
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        

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
    

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*4, kernel_size=(3, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels*4),
        )
        window_len = (window_len - 3 + 2) // 3 + 1
        window_len = (window_len - 3 + 2) // 1 + 1
        
        
        self.fc = nn.Sequential(
            nn.Linear(window_len*(out_channels*4)*num_channels, num_classes)
        )
    def get_feature(self):
        return self.feature
    def forward(self, x, meta_loss=None):
        x = x.unsqueeze(1)
        out1 = self.Block1(x, meta_loss)
        y1 = self.shortcut1(x, meta_loss)
        out = y1 + out1

        out2 = self.Block2(out, meta_loss)
        y2 = self.shortcut2(out, meta_loss)
        out = y2 + out2

        out3 = self.Block3(out, meta_loss)
        y3 = self.shortcut3(out, meta_loss)
        out = y3 + out3
        self.feature = out 
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda()
        return out   
if __name__ == "__main__":
    import torch
    torch.cuda.set_device("cuda:1")
    window_len,num_channels, num_classes = 128,113, 6
    model = ResnetMeta(window_len, num_channels, num_classes).cuda()
    a = torch.rand(16, window_len, num_channels).cuda()
    
    print(model(a))
    pass