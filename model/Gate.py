from torch import nn
import torch
# TODO 考虑个体的统计特征
    
class Gate3(nn.Module):

    def __init__(self, window_len, num_channels, num_submodels, num_hiddens=16):
        super(Gate3, self).__init__()
        self.num_classes = num_submodels
        self.window_len = window_len
        self.num_channels = num_channels
        self.num_hiddens = num_hiddens
        self.mlp1 = nn.Linear(in_features=4*num_channels, out_features=num_hiddens)
        self.mlp2 = nn.Linear(in_features=num_hiddens, out_features=num_hiddens)
        self.mlp3 = nn.Linear(in_features=num_hiddens,out_features=num_submodels)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(num_hiddens)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        min_x = torch.min(x, dim=1)[0].unsqueeze(1)
        max_x = torch.max(x, dim=1)[0].unsqueeze(1)
        mean_x = torch.mean(x, dim=1).unsqueeze(1)
        std_mean_x = torch.std_mean(x, dim=1)[0].unsqueeze(1)
        
        x = torch.cat([min_x, max_x, mean_x, std_mean_x], dim=1)
        x = x.view(x.shape[0], -1)

        x = self.mlp1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        #x = self.bn(x)
        x = self.mlp2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        #x = self.bn(x)
        out = self.mlp3(x)
        out = nn.functional.softmax(out, dim=1)
        return out
class Gate4(nn.Module):

    def __init__(self, window_len, num_channels, num_submodels, kernel_size=5):
        super(Gate4, self).__init__()
        self.num_channels = num_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(kernel_size, 1), stride=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(kernel_size, 1), stride=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(kernel_size, 1), stride=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        window_len = (window_len - kernel_size) // 2 + 1
        window_len = (window_len - kernel_size) // 2 + 1
        window_len = (window_len - kernel_size) // 2 + 1
        self.fc = nn.Sequential(
            nn.Linear(in_features=256*window_len*num_channels,out_features=512),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=512,out_features=num_submodels)
        )

    
    def forward(self, x):
        
        min_x = torch.min(x, dim=1)[0].unsqueeze(1)
        max_x = torch.max(x, dim=1)[0].unsqueeze(1)
        mean_x = torch.mean(x, dim=1).unsqueeze(1)
        std_mean_x = torch.std_mean(x, dim=1)[0].unsqueeze(1)
        # [64, 1, 4, 114]
        x = torch.cat([min_x, max_x, mean_x, std_mean_x], dim=1).unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        out = nn.functional.softmax(x, dim=1)
        return out
if __name__ == '__main__':
    model = Gate4(32, 51, 7).cuda()
    a = torch.rand(12,32,51).cuda()
    b = model.forward(a)