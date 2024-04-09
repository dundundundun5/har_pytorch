from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

# 生成时序的基本编码 此处用于分类不在乎未来，未来长度为0
def linspace(backcast_length: int, forecast_length: int, centered: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if centered:
        norm = max(backcast_length, forecast_length)
        start = -backcast_length
        stop = forecast_length - 1
    else:
        norm = backcast_length + forecast_length
        start = 0
        stop = backcast_length + forecast_length - 1
    lin_space = np.linspace(start / norm, stop / norm, backcast_length + forecast_length, dtype=np.float32)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class BaseBlock(nn.Module):
    def __init__(self, hidden, dim, block_type, length, out_channel, in_channel):
        super().__init__()
        self.block_type = block_type
        # basic_fc 四层全连接 单通道处理
        self.basic_fc = nn.Sequential(*[
            nn.Conv2d(length, hidden, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU()])

        # dim_fc 约束编码 和多项式or三角函数相乘
        self.dim_fc = nn.Sequential(*[nn.Conv2d(hidden, dim, kernel_size=1, stride=1, bias=False), nn.Tanh()])


        # trend 多项式约束
        backcast_linspace, forecast_linspace = linspace(length, 0, centered=False)
        norm = np.sqrt(1 / dim)
        coefficients = torch.tensor(np.array([backcast_linspace**i for i in range(dim)]), dtype=torch.float32)
        self.register_buffer("Trend_backcast", coefficients * norm)

        # seasonal 三角函数约束
        p1, p2 = (dim // 2, dim // 2) if dim % 2 == 0 else (dim // 2, dim // 2 + 1)
        s1_b = torch.tensor(np.array([np.cos(2 * np.pi * i * backcast_linspace) for i in np.linspace(0, length, p1)]), dtype=torch.float32)
        s2_b = torch.tensor(np.array([np.sin(2 * np.pi * i * backcast_linspace) for i in np.linspace(0, length, p2)]), dtype=torch.float32)
        self.register_buffer("Seasonal_backcast", torch.cat([s1_b, s2_b]))

        # generic 一层简易线性映射
        self.generic_fc = nn.Sequential(*[nn.Conv2d(dim, hidden, kernel_size=1, stride=1, bias=True), nn.BatchNorm2d(hidden), nn.ReLU(), nn.Conv2d(hidden, length, kernel_size=1, stride=1, bias=True)])

        # classify_fc 此block的分类模块，使用dim的隐向量进行分类
        self.classify_fc = nn.Sequential(*[
            nn.Linear((dim+4) * in_channel, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), 
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), 
            nn.Linear(hidden, out_channel)])

        
    def get_stat_and_norm(self, x):
        mean = torch.mean(x, dim=2, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False)).detach() + 1e-5
        x = x - mean
        x = x / stdev
        return x, mean, stdev

    def get_stat_and_minmaxnorm(self, x):
        min_x = torch.min(x, dim=2)[0].unsqueeze(2)
        max_x = torch.max(x, dim=2)[0].unsqueeze(2)
        x = x - min_x
        x = x / (max_x-min_x)
        return x, min_x, max_x

    def get_stat(self, x):
        mean = torch.mean(x, dim=2, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False)).detach() + 1e-5
        max_value = torch.max(x, dim=2)[0].unsqueeze(2)
        min_value = torch.min(x, dim=2)[0].unsqueeze(2)
        return mean, stdev, max_value, min_value

    def denorm(self, x, mean, stdev):
        x = x * stdev
        x = x + mean
        return x

    def minmaxdenorm(self, x, min_x, max_x):
        x = x * (max_x-min_x)
        x = x + min_x
        return x

    def forward(self, x, model_type=None):
        # 这里的x尺寸为batch, 通道 , 时序
        mean, stdev, max_value, min_value = self.get_stat(x)
        x, mean, stdev = self.get_stat_and_norm(x)
        # x, min_value, max_value = self.get_stat_and_minmaxnorm(x)
        # 这里的x尺寸为batch , 时序, 通道
        x = x.transpose(1, 2)
        # 拓展一个维度 用1*1卷积实现分通道的处理，即单轴的时序处理,batch , 时序, 通道 , 1
        x = x.unsqueeze(-1)
        # 1*1卷积处理为 batch，hidden，通道,1
        x = self.basic_fc(x)
        # 生成dim batch，dim,通道,1
        dim = self.dim_fc(x)
        back = None
        # 生成back部分 batch，通道，dim * dim，时序 -> batch，通道，时序
        if (self.block_type == 'trend'):
            back = dim.squeeze(-1).transpose(1, 2).matmul(self.Trend_backcast)
        if (self.block_type == 'seasonal'):
            back = dim.squeeze(-1).transpose(1, 2).matmul(self.Seasonal_backcast)
        if (self.block_type == 'generic'):
            back = self.generic_fc(dim).squeeze(-1).transpose(1, 2)
        # 使用dim进行分类
        back = self.denorm(back, mean, stdev)
        # back = self.minmaxdenorm(back, min_value, max_value)
        predict = self.classify_fc(torch.cat([dim.view(dim.shape[0], -1),mean.view(dim.shape[0], -1),stdev.view(dim.shape[0], -1),min_value.view(dim.shape[0], -1),max_value.view(dim.shape[0], -1)], 1))
        return back, predict, max_value, min_value


class MTSDNet(nn.Module):
    def __init__(self, out_channel, in_channel, hidden, dim, structure_str,length):
        super(MTSDNet, self).__init__()
        # 重构后的NBEATCS代码
        # hidden为中间层的节点尺寸
        # dim为多项式约束以及三角函数约束的复杂性
        # structure_str 表明模型结构的构成 趋势-周期-残差
        # 同一名称指代对象全局统一
        # 此版本使用多模态的统计视图信息融合 
        # 此版本使用每一层的逐层的标准化  暂且不使用反标准化
        self.net_block = nn.ModuleList()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden = hidden
        self.dim = dim  # 一般而言 dim至少为3，周期 dim=2 为全0和全1 趋势dim=1 为全1

        # 根据设计结构搭建层次结构
        block_type = ['trend', 'seasonal', 'generic']
        for j in structure_str:
            if (j == 't'):
                self.net_block.append(BaseBlock(hidden=hidden, dim=dim, block_type=block_type[0], length=length, out_channel=out_channel, in_channel=in_channel))
            if (j == 's'):
                self.net_block.append(BaseBlock(hidden=hidden, dim=dim, block_type=block_type[1], length=length, out_channel=out_channel, in_channel=in_channel))
            if (j == 'g'):
                self.net_block.append(BaseBlock(hidden=hidden, dim=dim, block_type=block_type[2], length=length, out_channel=out_channel, in_channel=in_channel))
        self.w = nn.Parameter(torch.ones(len(self.net_block)))

    def forward(self, x, model_type=None):
        x = x.transpose(1, 2)
        x = x.squeeze(-1)
        backcast = x
        forecast = []
        for i, block in enumerate(self.net_block):
            backcast_block, forecast_block, max_value, min_value = block(backcast)
            # backcast = backcast / torch.clamp(backcast_block, 0.5, 2)
            backcast = backcast - torch.clamp(backcast_block, min_value-0.1*torch.abs(min_value), max_value+0.1*torch.abs(max_value))
            # backcast = backcast - backcast_block
            forecast.append(forecast_block)

        weight = torch.exp(self.w) / torch.sum(torch.exp(self.w))
        out = forecast[0]*weight[0]
        for i in range(len(self.net_block) - 1):
            out += forecast[i+1]*weight[i+1]
        return out

    def get_stat_and_norm(self, x):
        mean = torch.mean(x, dim=2, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False)).detach()
        x = x - mean
        x = x / stdev
        return x, mean, stdev


if __name__ == "__main__":
    model = MTSDNet(out_channel=12, in_channel=3, hidden=2, dim=2, structure_str='tsg', length=5)
    a = torch.rand(4, 5, 3)
    b = model.forward(a)
    print(b.shape)