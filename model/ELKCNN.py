import torch
import torch.nn as nn
import sys
import os

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=in_channels, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

def get_bn(channels):
    return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        if type(kernel_size) == tuple:
            padding = (kernel_size[0] // 2,kernel_size[1] // 2)
        elif type(kernel_size) == int:
            padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        if type(kernel_size) == tuple:
            padding = (kernel_size[0] // 2,kernel_size[1] // 2)
        elif type(kernel_size) == int:
            padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    # 1: conv.weight * (bn.scale / (bn.variance + 0.00001))
    # 2: 0 （conv_bias Conv2D(bias=False)) + bn.bias - bn.mean * bn.scale / (bn.variance + 0.00001)
    
    # bn = (x - bn.mean) / bn.std * bn.weight + bn.shift= x / bn.std * bn.weight 
    # (conv(bias=0)+bn).weight = conv.weight * bn.weight / bn.std
    # -bn.mean/bn.std *bn.weight + bn.shift
    
    
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
   
        if type(kernel_size) == tuple:
            padding = (kernel_size[0] // 2,kernel_size[1] // 2)
        elif type(kernel_size) == int:
            padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2 if type(small_kernel) == int else (small_kernel[0]//2,small_kernel[1]//2), groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'): 
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
       
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            pad_number = (self.kernel_size[0] - self.small_kernel[0])// 2
            eq_k += nn.functional.pad(small_k.permute(0,1,3,2), (pad_number, pad_number)).permute(0,1,3,2)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ELK(nn.Module):
    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel,small_kernel_merged=False):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.pw2 = conv_bn(dw_channels, dw_channels, 1, 1, 0, groups=1)
        self.prelkb_bn = get_bn(dw_channels)
       

        self.short = nn.Sequential()
        if (in_channels != dw_channels):
            self.short = nn.Sequential(
                 nn.Conv2d(in_channels,dw_channels,1),
                 nn.ReLU(),
            )


    def forward(self, x):
        out = self.pw1(x)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)     
        return  self.short(x) + (out)


class ELK_CNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ELK_CNN, self).__init__()
       
        self.layer = nn.Sequential(
            nn.Conv2d(1,64, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            ELK(64,128,(31,1),(3,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, num_channels))
        self.fc = nn.Linear(256*num_channels, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


if __name__ == '__main__':
    
    a = torch.rand(64, 128, 113)
    model = ELK_CNN(113 , 18)
    
    print(model(a))
    b = torch.rand(64, 128, 113)
    model.structural_reparam()
    print(model(b))

    
