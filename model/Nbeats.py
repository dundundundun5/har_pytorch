import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def linspace(backcast_length):
    return np.linspace(0, 1, backcast_length, dtype=np.float32)


# 所有linear层改成conv1x1，从而适应多通道
class NbeatsBasicBlock(nn.Module):
    def __init__(self, units, thetas_dim, backcast_length, num_channels, num_block_layers=4
                 , dropout=0.1):
        super(NbeatsBasicBlock, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.num_channels = num_channels
        # 在最开始把X映射成x张特征图
        fc_stack = [
            nn.Conv2d(in_channels=1, out_channels=units, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=units),
            nn.ReLU()
        ]
        for _ in range(num_block_layers - 1):
            fc_stack.extend([nn.Dropout(dropout)])
            fc_stack.extend([nn.Conv2d(in_channels=units, out_channels=units, kernel_size=(1, 1))])
            fc_stack.extend([nn.ReLU()])

        self.fc = fc_stack
        self.theta_b_fc = nn.Conv2d(units, thetas_dim, kernel_size=(1, 1), bias=False)
        # 用于把输出塑形为原来输入的样子
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=thetas_dim, out_channels=1, kernel_size=(1, 1)),
            nn.Tanh(),
        )

    def forward(self, x):
        for layer in self.fc:
            x = layer.cuda()(x)
        return x


class NbeatsSeasonalBlock(NbeatsBasicBlock):
    def __init__(self, units, thetas_dim,
                 backcast_length, num_channels, num_block_layers=4, dropout=0.1):
        # theta_dim是在施加约束项之前，进行维度压缩的步骤
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            num_channels=num_channels,
            dropout=dropout,
        )
        backcast_linspace = linspace(backcast_length)
        # 0~H/2 cos  H/2+1~H sin, 故p1p2分别表示当总长是偶数或奇数时
        # 两段谐振因子的项个数，以及起始终止位置
        p1, p2 = (thetas_dim // 2, thetas_dim // 2) \
            if thetas_dim % 2 == 0 \
            else (thetas_dim // 2, thetas_dim // 2 + 1)
        # 生成[0,backcast_length]中p1个长度序列
        frequency1 = np.linspace(0, backcast_length, p1)
        frequency2 = np.linspace(0, backcast_length, p2)
        s1_b = torch.tensor(
            np.array([np.cos(2 * np.pi * i * backcast_linspace) for i in frequency1]), dtype=torch.float32
        )

        s2_b = torch.tensor(
            np.array([np.sin(2 * np.pi * i * backcast_linspace) for i in frequency2]), dtype=torch.float32
        )
        # 在模型缓冲池存变量，不会被视为模型参数
        # S_backcast是个方阵
        temp = torch.cat([s1_b, s2_b])
        self.S_backcast = temp.unsqueeze(0).unsqueeze(0).cuda()

    def forward(self, x):
        x = super().forward(x)
        a: torch.Tensor = self.theta_b_fc(x)
        # [1,1,length,length] x [ba]
        backcast = torch.matmul(self.S_backcast, a)
        backcast = self.final(backcast)
        return backcast


class NbeatsTrendBlock(NbeatsBasicBlock):
    def __init__(self, units, thetas_dim, backcast_length, num_channels, num_block_layers=4, dropout=0.1):
        super().__init__(units=units, thetas_dim=thetas_dim,
                         backcast_length=backcast_length, num_channels=num_channels,
                         num_block_layers=num_block_layers, dropout=dropout)
        backcast_linspace = linspace(backcast_length)
        # 源代码中这句话是要把预测区间的数值缩放norm = np.sqrt(forecast_length / thetas_dim)
        # coefficients也是一个方阵型
        coefficients = torch.tensor(np.array([backcast_linspace ** i for i in range(thetas_dim)]), dtype=torch.float32)
        self.T_backcast = coefficients.unsqueeze(0).unsqueeze(0).cuda()

    def forward(self, x):
        x = super().forward(x)
        a = self.theta_b_fc(x)
        backcast = torch.matmul(self.T_backcast, a)
        backcast = self.final(backcast)
        return backcast


class NbeatsGenericBlock(NbeatsBasicBlock):
    def __init__(self, units, thetas_dim,
                 backcast_length, num_channels, num_block_layers=4,
                 dropout=0.1):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            num_channels=num_channels,
            backcast_length=backcast_length,
            dropout=dropout,
        )
        self.backcast_fc = nn.Sequential(
            nn.Conv2d(in_channels=thetas_dim, out_channels=backcast_length, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=backcast_length, out_channels=1, kernel_size=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = super().forward(x)
        theta_b = self.theta_b_fc(x)
        backcast = F.relu(theta_b)
        backcast = self.backcast_fc(backcast)
        return backcast


class Nbeats(nn.Module):
    def __init__(self, units, thetas_dim, backcast_length, num_channels, num_classes, num_block_layers=4
                 , dropout=0.1, block_number=2, stack_number=None):
        # 为以后可扩展性做准备
        super().__init__()
        self.num_classes = num_classes
        self.block_number = block_number
        self.stack_number = stack_number # not used
        self.backcast_length = backcast_length
        self.num_channels = num_channels

        s,t,g = [],[],[]
        for _ in range(block_number):
            s.append(NbeatsSeasonalBlock(units, thetas_dim, backcast_length,
                                            num_channels, num_block_layers, dropout))
            t.append(NbeatsTrendBlock(units, thetas_dim, backcast_length,
                                      num_channels, num_block_layers, dropout))
            g.append(NbeatsGenericBlock(units, thetas_dim, backcast_length,
                                          num_channels, num_block_layers, dropout))
        self.seasonal = nn.Sequential(*s)
        self.trend = nn.Sequential(*t)
        self.generic = nn.Sequential(*g)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backcast_length * num_channels, self.num_classes),
            nn.Dropout(0.5),
            nn.Linear(self.num_classes, self.num_classes),
        )
        # 3 表示3个stack
        self.weight = nn.Parameter(torch.ones(3), requires_grad=True).cuda()

    def forward(self, x):
        # TODO 多加层数
        x = x.unsqueeze(1)
        seasonal_output, trend_output, generic_output = float(0),float(0),float(0)
        for layer in self.seasonal:
            seasonal_output += layer(x)
            x = x - seasonal_output
        for layer in self.trend:
            trend_output += layer(x)
            x = x - trend_output
        for layer in self.generic:
            generic_output += layer(x)
            x = x - generic_output
        b = seasonal_output.shape[0]
        outputs = [seasonal_output, trend_output, generic_output]
        weight_sum = torch.sum(torch.exp(self.weight), dim=0)
        total_output = torch.zeros(seasonal_output.shape).cuda()
        # 3表示有3个stack
        for i in range(3):
            w = torch.exp(self.weight[i]) / weight_sum
            total_output = total_output + w * outputs[i]
        total_output = total_output.view(b, -1)
        total_output = self.fc(total_output)
        # total_output = F.softmax(total_output, dim=1)
        return total_output


if __name__ == '__main__':
    # [batch_size, sliding_window_length, num_channels]
    x: torch.Tensor = torch.rand(9, 15, 12).cuda()
    # [batch_size, in_  channels, sliding_window_length, num_channels]
    x = x.reshape(9, -1, 15, 12)  # permute仅供测试，改成多通道后要注意batch_size
    s = Nbeats(units=10, thetas_dim=15, num_channels=12, backcast_length=15, num_classes=5).cuda()
    o = s.forward(x)
    print(o)
