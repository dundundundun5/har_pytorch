import torch
import torch.nn as nn




def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss
class DeepCoralCNN(nn.Module):

    def __init__(self, window_len, num_channels,num_classes, out_channels=64):
        super(DeepCoralCNN, self).__init__()
        conv1 = nn.Conv2d(1, out_channels, (3, 1), (3, 1), padding=(1, 0))
        
        window_len = (window_len - 3 + 2) // 3 + 1
        
        conv2 = nn.Conv2d(out_channels, out_channels*2, (3, 1), (3, 1), padding=(1, 0))
        
        window_len = (window_len - 3 + 2) // 3 + 1
        
        conv3 = nn.Conv2d(out_channels*2, out_channels*4, (3, 1), (3, 1), padding=(1, 0))
        
        window_len = (window_len - 3 + 2) // 3 + 1
        
        self.sharedNet = nn.Sequential(
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
        self.fc = nn.Sequential(
            nn.Linear(in_features=window_len*(out_channels*4)*num_channels, out_features=num_classes)
        )          
        
                        

    def forward(self, source, target):
        source, target = source.unsqueeze(1), target.unsqueeze(1)
        source = self.sharedNet(source)
        source = source.reshape(source.shape[0], -1)
        source = self.fc(source)

        target = self.sharedNet(target)
        target = target.reshape(target.shape[0], -1)
        target = self.fc(target)
    
        
        return source, target
if __name__ == "__main__":

    l,c, cl = 128,12, 14
    model = DeepCoralCNN(l,c,cl)
    source = torch.rand(16 ,l, c)
    target = torch.rand(16, l, c)
    out1, out2 = model(source, target)

    coral_loss = CORAL(out1, out2)
    print(coral_loss)
    # total = coral_loss * (epoch+1 / epochs) + criterion 
    pass