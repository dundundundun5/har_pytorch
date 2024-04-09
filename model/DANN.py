import torch
import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.alpha = 1

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None    
class DANNCNN(nn.Module):

    def __init__(self, window_len, num_channels,num_classes, out_channels=64):
        super(DANNCNN, self).__init__()
        conv1 = nn.Conv2d(1, out_channels, (3, 1), (3, 1), padding=(1, 0))
        
        window_len = (window_len - 3 + 2) // 3 + 1
        
        conv2 = nn.Conv2d(out_channels, out_channels*2, (3, 1), (3, 1), padding=(1, 0))
        
        window_len = (window_len - 3 + 2) // 3 + 1
        
        conv3 = nn.Conv2d(out_channels*2, out_channels*4, (3, 1), (3, 1), padding=(1, 0))
        
        window_len = (window_len - 3 + 2) // 3 + 1
        
        self.conv_module = nn.Sequential(
            conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            

            conv2,
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True),
            

            conv3,
            nn.BatchNorm2d(out_channels*4),
            nn.ReLU(inplace=True),
            
        )
        # 计算输入特征数
        # 参数化输出类别
        self.class_classifier = nn.Sequential(
            nn.Linear(in_features=window_len*(out_channels*4)*num_channels, out_features=num_classes)
        )          
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features=window_len*(out_channels*4)*num_channels, out_features=2)
        )                  

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_module(x) # [B,1,L,C] -> [B,256,,C]
        x = x.reshape(x.shape[0], -1)
        reverse_x = ReverseLayerF.apply(x)
        out_class, out_domain = self.class_classifier(x), self.domain_classifier(reverse_x)
        
        return out_class, out_domain
if __name__ == "__main__":

    l,c, cl = 128,12, 14
    model = DANNCNN(l,c,cl)
    a = torch.rand(16 ,l, c)
    
    print(model(a)[0].shape)
    print(model(a)[1].shape)
    pass