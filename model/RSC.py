import numpy as np
import torch
from torch import nn
import torch.autograd as autograd
import torch.nn.functional as F

class RSCCNN(nn.Module):
    def __init__(self, window_len, num_channels,num_classes, out_channels=64):
        super(RSCCNN, self).__init__()
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
        self.rsc_f_drop_factor, self.rsc_b_drop_factor = 0.5, 0.5
        self.drop_f = (1 - self.rsc_f_drop_factor) * 100
        self.drop_b = (1 - self.rsc_b_drop_factor) * 100
    

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        all_f = self.sharedNet(x)
        all_f = all_f.reshape(all_f.shape[0], -1)
        all_p = self.fc(all_f)
        return all_p, all_f

if __name__ == "__main__":

    l,c, cl = 128,12, 2
    model = RSCCNN(l,c,cl).cuda()
    train_X = torch.rand(16 ,l, c).float().cuda()
    train_y = torch.ones(16, 1).long().cuda()
    train_y = train_y.squeeze(1)
    all_p, all_f = model(train_X)
    # Equation (1): compute gradients with respect to representation
    all_o = torch.nn.functional.one_hot(train_y, cl)
    all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

    # Equation (2): compute top-gradient-percentile mask
    percentiles = np.percentile(all_g.cpu(), model.drop_f, axis=1)
    percentiles = torch.Tensor(percentiles)
    percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
    mask_f = all_g.lt(percentiles.cuda()).float()

    # Equation (3): mute top-gradient-percentile activations
    all_f_muted = all_f * mask_f

    # Equation (4): compute muted predictions
    all_p_muted = model.fc(all_f_muted)

    # Section 3.3: Batch Percentage
    all_s = F.softmax(all_p, dim=1)
    all_s_muted = F.softmax(all_p_muted, dim=1)
    changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
    percentile = np.percentile(changes.detach().cpu(), model.drop_b)
    mask_b = changes.lt(percentile).float().view(-1, 1)
    mask = torch.logical_or(mask_f, mask_b).float()

    # Equations (3) and (4) again, this time mutting over examples
    predict_y = model.fc(all_f * mask)
    print(predict_y.shape)