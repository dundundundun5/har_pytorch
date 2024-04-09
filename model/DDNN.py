# encoding=utf-8
"""
    Created on 2018/12/8
    @author: Hangwei Qian
    [batch_size, sliding_window_length, num_channels]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# H = (h -k +2xpad)//s + 1
# [B,L,C]输入
class DDNN(nn.Module):
    def __init__(self, n_lstm_hidden, n_lstm_layer, d_AE, sliding_window_length, num_channels, num_classes, out_channels=1024, fc_hiddens=1000):
        super(DDNN, self).__init__()
        # parameter part
        self.n_lstm_hidden = n_lstm_hidden
        self.n_lstm_layer = n_lstm_layer

        self.n_feature = num_channels
        self.len_sw = sliding_window_length
        self.n_class = num_classes
        self.d_AE = d_AE

        # autoencoder part
        self.encoder = nn.Sequential(
            nn.Linear(self.n_feature * self.len_sw, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, self.d_AE))
        self.decoder = nn.Sequential(
            nn.Linear(self.d_AE, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, self.n_feature * self.len_sw),
            nn.Tanh())
        # rnn part
        self.lstm = nn.LSTM(self.n_feature, self.n_lstm_hidden, self.n_lstm_layer, batch_first=True)
        self.lstm_spatial = nn.LSTM(self.len_sw, self.n_lstm_hidden, self.n_lstm_layer, batch_first=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=out_channels, kernel_size=(1, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels // 8, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels // 8, out_channels=out_channels // 16, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        len_after_conv1 = (self.len_sw - 5)//1 + 1
        len_after_conv1 = (len_after_conv1 - 2) // 2 + 1
        len_after_conv2 = (len_after_conv1 - 3) // 1 + 1
        len_after_conv2 = (len_after_conv2 - 2) // 2 + 1
        len_after_conv3 = (len_after_conv2 - 1) // 1 + 1
        len_after_conv3 = (len_after_conv3 - 2) // 2 + 1
        len_after_conv4 = (len_after_conv3 - 1) // 1 + 1
        len_after_conv4 = (len_after_conv4 - 2) // 2 + 1
        self.res_len = len_after_conv4

        ## fc part after concat of three networks
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=(2*self.n_lstm_hidden + self.d_AE + 64 * self.res_len), out_features=fc_hiddens),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fc_hiddens, out_features=fc_hiddens//2),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=fc_hiddens//2, out_features=self.n_class)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # added
        out_encoder = self.encoder(x.reshape(x.size(0), -1))
        out_decoder = self.decoder(out_encoder)

        out_rnn, _ = self.lstm(x.permute(0, 2, 1)) # (64, 100, 9)
        out_rnn = out_rnn[:, -1, :]

        out_rnn_spatial, _ = self.lstm_spatial(x.reshape(x.shape[0], self.n_feature, -1))  # (64, 9, 100)
        out_rnn_spatial = out_rnn_spatial[:, -1, :]

        out_conv1 = self.conv1(x.reshape(-1, x.shape[1], 1, x.shape[2]))
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv4 = out_conv4.reshape(-1, out_conv4.shape[1] * out_conv4.shape[3])

        out_combined = torch.cat((out_encoder.view(out_encoder.size(0), -1), out_rnn.view(out_rnn.size(0), -1),
                                  out_rnn_spatial.view(out_rnn_spatial.size(0), -1),
                                  out_conv4.view(out_conv3.size(0), -1)), dim=1)  # (64,184)
        out_combined = self.fc1(out_combined)
        out_combined = self.fc2(out_combined)
        out_combined = self.fc3(out_combined)
        # out_combined = F.softmax(out_combined, dim=1)
        # return out_combined, out_decoder # origin code
        return out_combined
