import torch
from torch import nn
import torch.nn.functional as F

class DeepConvLSTM(nn.Module):
    def __init__(self, num_classes, num_channels, lstm_hiddens, num_filters):
        super(DeepConvLSTM, self).__init__()
        self.model_name = 'deepconvlstm'
        self.lstm_hiddens = lstm_hiddens
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.num_channels = num_channels
        # 模型部分
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(5, 1))
        self.conv2 = nn.Conv2d(num_filters, num_filters, (5, 1)) # stride = 1
        self.conv3 = nn.Conv2d(num_filters, num_filters, (5, 1))
        self.conv4 = nn.Conv2d(num_filters, num_filters, (5, 1))
        self.lstm1 = nn.LSTM(input_size=num_filters, hidden_size=lstm_hiddens, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hiddens, hidden_size=lstm_hiddens, num_layers=1, batch_first=True)
        self.out = nn.Linear(self.lstm_hiddens, num_classes)
    def forward(self, x):
        """
         
        由于Batch_size 为100，因此对于卷积层而言，其tensor size 的变化为：
        100 * 1 * 24 * 113 --> 100 * 64 * 20 * 113 --> 100 * 64 * 16 * 113–> 100 * 64 * 12 * 113–> 100 * 64 * 8 * 113

        然后为了与Dense层(LSTM层)相连接，需要reshape 一次，转成大小为 100 * 8 * 7232的tensor
        LSTM层的隐藏单元为128个。
        输出层的激活函数为softmax，其它层则为relu。

        """
        x = x.unsqueeze(1)
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x))) #
        # 特征图的大小和卷积核（大小，步长）的关系
        b = x.shape[0]
        x = x.reshape(b, -1, self.num_filters)
        # x = F.dropout(x, p=0.5)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.out(x)
        x = x[:,-1,:]
        return x


if __name__ == '__main__':
    a = torch.rand(9, 1,24, 12)
    net = DeepConvLSTM(5, 12,128,64)
    b = net.forward(a)
    print(b)
