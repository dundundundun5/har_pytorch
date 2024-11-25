# TODO TCCSNet复现
from torch import nn
import torch
class SelfAttentionBlock(nn.Module):
    def __init__(self, num_channels) -> None:
        super(SelfAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=num_channels)
        # multihead attention + dropout 
        num_heads = 4
        if num_channels // 2 != 0:
            num_heads = 1 

        self.msa = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, dropout=0.3, batch_first=True) 
        self.ln2 = nn.LayerNorm(normalized_shape=num_channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=num_channels),
            nn.ReLU(True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add skip-connect
        ln1_x = self.ln1(x)
        out = self.msa(ln1_x, ln1_x, ln1_x)[0] + x
        # add skip-connect
        out = self.ffn(self.ln2(out)) + out
        return out
    

class CSNet(nn.Module):
    def __init__(self, window_len, num_channels, num_classes, cuda_device=None, filter = 32) -> None:
        super(CSNet, self).__init__()
        self.num_channels  = num_channels
        self.filter = filter
        self.conv_block1 = nn.Sequential(
            # Conv Block k=3 F=32,64
            nn.Conv2d(in_channels=1, out_channels=filter, kernel_size=(3, 1)),
            nn.BatchNorm2d(filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filter, out_channels=2*filter, kernel_size=(3, 1)),
            nn.BatchNorm2d(2*filter),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1)),
        )
        window_len = window_len - 4
        # sin-cos position encoding
        temp = torch.zeros(512, window_len)
        position = torch.arange(0, 512, dtype=torch.float).unsqueeze(1)
        div2 = torch.pow(torch.tensor(10000.0), torch.arange(0, window_len, 2).float() / window_len)
        div1 = torch.pow(torch.tensor(10000.0), torch.arange(1, window_len, 2).float() / window_len)
        temp[:, 0::2] = torch.sin(position * div2)
        temp[:, 1::2] = torch.cos(position * div1)
        
        self.positional_embedding = temp[:num_channels, :].transpose(0, 1)
        if cuda_device is not None:
            torch.cuda.set_device(f"cuda:{cuda_device}")
            self.positional_embedding = self.positional_embedding.cuda()
        self.sa1 = SelfAttentionBlock(num_channels * 2*filter)
        self.sa2 = SelfAttentionBlock(num_channels * 2*filter)
        self.conv_block2 = nn.Sequential(
            # Conv Block k=3 F=64
            nn.Conv2d(in_channels=2*filter, out_channels=2*filter, kernel_size=(3, 1)),
            nn.BatchNorm2d(2*filter),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1)),
        )
        window_len = window_len - 2
        self.fc = nn.Sequential(
            nn.Linear(in_features=2*filter*window_len*num_channels, out_features=32),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(in_features=32, out_features=num_classes),
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block1(x)
        
        x = x + self.positional_embedding
        x = x.view(x.shape[0], -1, self.num_channels * 2*self.filter)
        x = self.sa1(x)
        x = self.sa2(x)
        x = x.view(x.shape[0], 2*self.filter, -1, self.num_channels)
        x = self.conv_block2(x)
        
        x = x.reshape(x.shape[0], -1)
        
        out = self.fc(x)      
        return out
if __name__ == '__main__':
    cuda_device = 2
    torch.cuda.set_device(f"cuda:{cuda_device}")
    x = torch.rand(32, 64, 113).cuda()
    model = CSNet(64, 113, 6, cuda_device=cuda_device).cuda()
    
    print(model(x).shape)