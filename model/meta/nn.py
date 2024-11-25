from torch import nn
import torch.nn.functional as F
from torch import autograd
meta_lr = 0.001
import torch

def grad_weight_n_bias(weight, bias, meta_loss):
    if meta_loss is not None:
        grad_weight = autograd.grad(meta_loss, weight, create_graph=True)[0]
        grad_bias = autograd.grad(meta_loss, bias, create_graph=True)[0]
        # 一步梯度下降
        weight = weight - meta_lr * grad_weight
        bias = bias - meta_lr * grad_bias
    return weight, bias

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x, meta_loss=None):
         weight, bias = grad_weight_n_bias(self.layer.weight, self.layer.bias, meta_loss)
         out = F.conv2d(x, weight, bias, self.layer.stride, self.layer.padding)
         return out
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias:bool=True) -> None:
        super(Linear, self).__init__()
        self.layer = nn.Linear(in_features, out_features, bias)
    def forward(self, x, meta_loss=None):
        weight, bias = grad_weight_n_bias(self.layer.weight, self.layer.bias, meta_loss)
        out = F.linear(x, weight, bias)
        return out
class ReLU(nn.Module):
    
    def __init__(self, inplace=True):
        super(ReLU, self).__init__()
        self.layer = nn.ReLU(inplace=inplace)
    def forward(self, x, meta_loss=None):
        out = self.layer(x)
        return out
class BatchNorm2d(nn.Module):
    
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__()
        self.layer = nn.BatchNorm2d(num_features)
    def forward(self, x, meta_loss=None):
        out = self.layer(x)
        return out
class Module(nn.Module):
    
    def __init__(self):
        super(Module, self).__init__()
    def forward(self, x, meta_loss=None):
        return x
class Sequential(nn.Sequential):
    
    def __init__(self, *args: nn.Module):
        super(Sequential, self).__init__(*args)
    def forward(self, x, meta_loss=None):
        for module in self:
            x = module(x, meta_loss)
        return x
class LayerNorm(nn.LayerNorm):
    
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__(normalized_shape)
    def forward(self, x, meta_loss=None):
        out = super(LayerNorm, self).forward(x)
        return out

if __name__ == '__main__':
    
    print("import sucessfully")
    print(f"meta_lr={meta_lr}")