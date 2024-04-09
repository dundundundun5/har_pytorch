"""
for source_loader in source_loaders:
        for batch_idx, (x, y, d) in enumerate(source_loader):
            x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)

            optimizer.zero_grad()
            false_optimizer.zero_grad()

            loss_origin, class_y_loss = model.loss_function(d, x, y)

            loss_false = model.loss_function_false(args, d, x, y)

            loss_origin.backward()
            optimizer.step()
            loss_false.backward()
            false_optimizer.step()
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
# decoder
# pzx
class px(nn.Module):
    def __init__(self, zd_dim, zy_dim, window_len,args):
        super(px, self).__init__()

        
        self.n_feature = args.num_channels
        self.fc1 = nn.Sequential(
            nn.Linear(zd_dim + zy_dim, 64*window_len, bias=False), 
            nn.BatchNorm1d(64*window_len), 
            nn.ReLU()
            )

        self.un1 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU()
        )

        self.un2 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=(1, 1)),
            nn.ReLU()
        )

        self.un3 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=(1, 1)),
            nn.ReLU()
        )

        self.un4 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=self.n_feature, kernel_size=(1, 5)),
            nn.ReLU()
        )

    def forward(self, zd, zy, idxs, sizes):
        zdzy = torch.cat([zd, zy], dim=-1)
        h = self.fc1(zdzy)
        h_shape = idxs[3].shape
        h = h.view(h_shape[0], h_shape[1], 1, -1)
        
        out_1 = self.un1(h, idxs[3], output_size=sizes[2])
        out_11 = self.deconv1(out_1)

        out_2 = self.un2(out_11, idxs[2], output_size=sizes[1])
        out_22 = self.deconv2(out_2)

        out_3 = self.un3(out_22, idxs[1], output_size=sizes[0])
        out_33 = self.deconv3(out_3)

        out_4 = self.un4(out_33, idxs[0])
        out_44 = self.deconv4(out_4)
        out = out_44.permute(0, 2, 3, 1)
        return out


class pzd(nn.Module):
    def __init__(self, d_dim, zd_dim):
        super(pzd, self).__init__()
        self.d_dim = d_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_dim, zd_dim, bias=False), 
            nn.BatchNorm1d(zd_dim),
            nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

    def forward(self, d):
        d_onehot = torch.zeros(d.shape[0], self.d_dim)
        for idx, val in enumerate(d):
            d_onehot[idx][val.item()] = 1
        d_onehot = d_onehot.cuda()
        hidden = self.fc1(d_onehot)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7
        return zd_loc, zd_scale


class pzy(nn.Module):
    def __init__(self, y_dim,zy_dim):
        super(pzy, self).__init__()

        self.y_dim = y_dim
        self.fc1 = nn.Sequential(
            nn.Linear(y_dim, zy_dim, bias=False), 
            nn.BatchNorm1d(zy_dim), 
            nn.ReLU()
            )
        self.fc21 = nn.Sequential(
            nn.Linear(zy_dim, zy_dim)
            )
        self.fc22 = nn.Sequential(
            nn.Linear(zy_dim, zy_dim), 
            nn.Softplus()
            )

    def forward(self, y):
        y_onehot = torch.zeros(y.shape[0], self.y_dim)
        for idx, val in enumerate(y):
            y_onehot[idx][val.item()] = 1

        y_onehot = y_onehot.cuda()

        hidden = self.fc1(y_onehot)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale


# Encoders
class qzd(nn.Module):
    def __init__(self, zd_dim, args):
        super(qzd, self).__init__()

        self.n_feature = args.num_channels
        self.window_len = args.sliding_window_length
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=1024, kernel_size=(1, 5)),
            nn.ReLU()
        )
        self.window_len = (self.window_len - 5) // 1 + 1
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.window_len = (self.window_len - 2) // 2 + 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.window_len = (self.window_len - 1) // 1 + 1
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.window_len = (self.window_len - 2) // 2 + 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.window_len = (self.window_len - 1) // 1 + 1
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.window_len = (self.window_len - 2) // 2 + 1
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.window_len = (self.window_len - 1) // 1 + 1
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.window_len = (self.window_len - 2) // 2 + 1
        self.fc11 = nn.Sequential(
            nn.Linear(64*self.window_len, zd_dim)
            )
        self.fc12 = nn.Sequential(
            nn.Linear(64*self.window_len, zd_dim), 
            nn.Softplus())



    def forward(self, x):
        x_img = x.float()
        x_img = x_img.permute(0,3,1,2)
        # x_img [B,num_channels, 1, window_len]
        out_conv1 = self.conv1(x_img)
        out1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4 = self.pool4(out_conv4)

        out = out4.reshape(out4.shape[0], -1)


        # zd [batch_size, zd_dim]
        zd_loc = self.fc11(out) 
        zd_scale = self.fc12(out) + 1e-7

        return zd_loc, zd_scale



class qzy(nn.Module):
    def __init__(self, zy_dim, args):
        super(qzy, self).__init__()
        self.n_feature = args.num_channels
        self.window_len = args.sliding_window_length
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=1024, kernel_size=(1, 5)),
            nn.ReLU()
        )
        self.window_len = (self.window_len - 5) // 1 + 1
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True)
        self.window_len = (self.window_len - 2) // 2 + 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.window_len = (self.window_len - 1) // 1 + 1
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True)
        self.window_len = (self.window_len - 2) // 2 + 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.window_len = (self.window_len - 1) // 1 + 1
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True)
        self.window_len = (self.window_len - 2) // 2 + 1
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.window_len = (self.window_len - 1) // 1 + 1
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True)
        self.window_len = (self.window_len - 2) // 2 + 1
        self.fc11 = nn.Sequential(
            nn.Linear(64*self.window_len, zy_dim)
            
            )
        self.fc12 = nn.Sequential(
            nn.Linear(64*self.window_len, zy_dim), 
            nn.Softplus()
            )

    def forward(self, x):

        x_img = x.float()
        x_img = x_img.view(-1, x_img.shape[3], 1, x_img.shape[2])
        # x_img->out_conv1->out1,idx1
        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(out4.shape[0], -1) # [batch_size, 64, 1, 经过卷积后的长度]
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()
        size4 = out4.size()

        zy_loc = self.fc11(out) # []
        zy_scale = self.fc12(out) + 1e-7 
        # [idx] 和 [size]分别是用于反池化的输入特征图索引和输入特征图大小
        return zy_loc, zy_scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]

# Auxiliary tasks
class qd(nn.Module):
    def __init__(self, d_dim, zd_dim):
        super(qd, self).__init__()
        self.fc1 = nn.Linear(zd_dim, d_dim)

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)
        return loc_d


class qy(nn.Module):
    def __init__(self,y_dim, zy_dim):
        super(qy, self).__init__()
        self.fc1 = nn.Linear(zy_dim, y_dim)

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)

        return loc_y


class GILE(nn.Module):
    def __init__(self, args):
        super(GILE, self).__init__()
        self.zd_dim = 50
        self.zy_dim = 50
        self.d_dim = args.n_domains # 用于训练的志愿者个数
        self.y_dim = args.num_classes

        

        self.qzd = qzd(self.zd_dim, args)
        
        self.qzy = qzy(self.zy_dim, args)

        self.qd = qd(self.d_dim, self.zd_dim)
        self.qy = qy(self.y_dim, self.zy_dim)

        self.px = px(self.zd_dim, self.zy_dim, self.qzd.window_len,args)
        self.pzd = pzd(self.d_dim,self.zd_dim)
        self.pzy = pzy(self.y_dim, self.zy_dim)
        self.aux_loss_multiplier_y = 1000 
        self.aux_loss_multiplier_d = 1000
        self.weight_true = 1000
        self.weight_false = 1000
        
        self.beta_d = 0.002
        self.beta_y = 10

    def forward(self, d, x, y):
        x = torch.unsqueeze(x, 1)

        d = d.long()
        y = y.long()

        # Encode
        # [batch_size, zd_dim]
        zd_q_loc, zd_q_scale = self.qzd(x)

        zy_q_loc, zy_q_scale, idxs_y, sizes_y = self.qzy(x)

        # Reparameterization trick
        # 通过标准正态分布构造正态分布 zd_q_loc为偏移 zd_q_scale为倍数
        qzd = Normal(zd_q_loc, zd_q_scale) 
        zd_q = qzd.rsample() # 生成和输入样本形状一样的重参数样本

        qzy = Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # 拼接-反卷积-反池化-拼接zd zy得到重建x
        x_recon = self.px(zd_q, zy_q, idxs_y, sizes_y)

        zd_p_loc, zd_p_scale = self.pzd(d)
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = Normal(zd_p_loc, zd_p_scale) # zd的分布，用于KL散度比较
        pzy = Normal(zy_p_loc, zy_p_scale) # zy的分布，用于KL散度比较

        # Auxiliary losses
        d_hat = self.qd(zd_q) # 从隐空间重构后的域标签张量
        y_hat = self.qy(zy_q) # 从隐空间重构后的活动标签张量

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzy, pzy, zy_q

    def loss_function_false(self, d, x, y=None):
        d = d.long()
        y = y.long()

        pred_d, pred_y, pred_d_false, pred_y_false = self.classifier(x)

        loss_classify_true =  self.weight_true * (F.cross_entropy(pred_d, d, reduction='sum') + F.cross_entropy(pred_y, y, reduction='sum'))
        loss_classify_false = self.weight_false * (F.cross_entropy(pred_d_false, d, reduction='sum') + F.cross_entropy(pred_y_false, y, reduction='sum'))

        loss = loss_classify_true - loss_classify_false

        loss.requires_grad = True

        return loss

    def loss_function(self, d, x, y=None):
        d = d.long()
        y = y.long()
        x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzy, pzy, zy_q = self.forward(d, x, y)

        x = torch.unsqueeze(x, 1)
        if x.shape[2] != x_recon.shape[2]:
            x = x[:,:,:x_recon.shape[2],:]
        CE_x = F.mse_loss(x_recon, x.float())

        zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))


        zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))
        CE_d = F.cross_entropy(d_hat, d, reduction='sum')
        CE_y = F.cross_entropy(y_hat, y, reduction='sum')

        return CE_x - self.beta_d * zd_p_minus_zd_q - self.beta_y * zy_p_minus_zy_q + self.aux_loss_multiplier_d * CE_d + self.aux_loss_multiplier_y * CE_y, CE_y

    def classifier(self, x):
        # got d_pred, y_pred, d_false_pred, y_false_pred
        with torch.no_grad():

            x = torch.unsqueeze(x, 1)

            zd_q_loc, _= self.qzd(x)
            zd = zd_q_loc
            alpha = F.softmax(self.qd(zd), dim=1)
            

            ind = torch.topk(alpha,1)[1]
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0) # 域标签变独热
   
            zy_q_loc, _, _, _= self.qzy.forward(x)
            zy = zy_q_loc
            alpha = F.softmax(self.qy(zy), dim=1)
            
            ind2 = torch.topk(alpha, 1)[1]
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind2, 1.0)

            alpha_y2d = F.softmax(self.qd(zy), dim=1)


            ind3 = torch.topk(alpha_y2d, 1)[1]
            d_false = x.new_zeros(alpha_y2d.size())
            d_false = d_false.scatter_(1, ind3, 1.0)

            alpha_d2y = F.softmax(self.qy(zd), dim=1)
            
            ind4 = torch.topk(alpha_d2y, 1)[1]

            # convert the digit(s) to one-hot tensor(s)
            y_false = x.new_zeros(alpha_d2y.size())
            y_false = y_false.scatter_(1, ind4, 1.0)

        return d, y, d_false, y_false



