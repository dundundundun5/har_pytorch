import torch
import torch.nn as nn
import torch.nn.functional as F


class InputLayer(nn.Module):
    def __init__(
            self,
            window_len: int,
            embed_dim: int,
            max_len=512,args=None
    ) -> None:
        super().__init__()
        self.sliding_window_length = window_len
        self.num_channels = embed_dim
        # positional embedding
        # [max_len, window_len]
        temp = torch.zeros(max_len, window_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div2 = torch.pow(torch.tensor(10000.0), torch.arange(0, window_len, 2).float() / window_len)
        div1 = torch.pow(torch.tensor(10000.0), torch.arange(1, window_len, 2).float() / window_len)
        temp[:, 0::2] = torch.sin(position * div2)
        temp[:, 1::2] = torch.cos(position * div1)
        self.args = args
        torch.cuda.set_device(f"cuda:{self.args.cuda_device}")
        self.positional_embedding = temp[:embed_dim, :].transpose(0, 1).cuda()
        # self.positional_embedding = nn.parameter.Parameter(
        #     data=torch.randn(1, 1, self.sliding_window_length, self.num_channels)
        # )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.cuda.set_device(f"cuda:{self.args.cuda_device}")
        # x (B, 1, L, W) self.pe (none,none, L, W) using broadcasting mechanism
        out = x + self.positional_embedding
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, nb_head: int, dropout: float) -> None:
        """

        Args:
            embed_dim: 传感器模态数
            nb_head: 注意力头数，必须能把embed_dim整除
            dropout: [0,1)
        """
        super().__init__()
        self.nb_head = nb_head if embed_dim % nb_head == 0 else 1
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // nb_head if embed_dim % nb_head == 0 else embed_dim 

        self.w_q = nn.Linear(
            in_features=self.embed_dim, out_features=self.embed_dim, bias=False
        )
        self.w_k = nn.Linear(
            in_features=self.embed_dim, out_features=self.embed_dim, bias=False
        )
        self.w_v = nn.Linear(
            in_features=self.embed_dim, out_features=self.embed_dim, bias=False
        )
        self.w_o = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        window_len = x.shape[1]
        # (B, L, S) -> (B, nb_head, C, head_dim)
        q = self.w_q(x)
        q = q.view(batch_size, self.nb_head, window_len, self.head_dim)
        # key
        k = self.w_k(x)
        k = k.view(batch_size, self.nb_head, window_len, self.head_dim)
        # value
        v = self.w_v(x)
        v = v.view(batch_size, self.nb_head, window_len, self.head_dim)

        # inner product
        # (B, nb_head, L, embed_dim) × (B, nb_head, embed_dim, L) -> (B, nb_head, L, L)
        dots = torch.matmul(q, k.transpose(2, 3)) / self.head_dim ** 0.5
        # softmax by columns
        # dim=3 eq dim=-1. dim=-1 applies softmax to the last dimension
        attn = F.softmax(dots, dim=3)
        # weighted
        # (B, nb_head, L, L) × (B, nb_head, L, embed_dim) -> (B, nb_head, L, embed_dim)
        out = torch.matmul(attn, v)
        # (B, nb_head, L, embed_dim) -> (B, L, nb_head, embed_dim) -> (B, L, S)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        out = self.w_o(out)
        out = out.unsqueeze(1)
        return out


class EncoderBlock(nn.Module):
    def __init__(
            self, embed_dim: int, nb_head: int,
            hidden_dim: int, dropout: float, use_official_msa=False
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape=embed_dim)
        if use_official_msa:
            self.msa = \
                torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nb_head,
                                            dropout=dropout, batch_first=True)
        else:
            self.msa = MultiHeadSelfAttention(
                embed_dim=embed_dim, nb_head=nb_head, dropout=dropout
            )
        self.ln2 = nn.LayerNorm(normalized_shape=embed_dim)
        # embed_dim > hidden_dim > embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add skip-connect
        out = self.msa(self.ln1(x)).squeeze(1) + x
        # add skip-connect
        out = self.mlp(self.ln2(out)) + out
        return out


class VisionTransformer(nn.Module):
    def __init__(self, num_classes: int, window_len: int, embed_dim: int,
                 num_blocks: int, nb_head: int, hidden_dim: int, dropout: float,args) -> None:
        """

        Args:
            num_classes: 活动类别数
            window_len: 滑窗长度
            embed_dim: 传感器轴数
            num_blocks: 编码器块数
            nb_head: 注意力头数
            hidden_dim: mlp隐藏层单元数
            dropout: 丢弃率 (0,1]
        """
        super().__init__()
        self.args = args
        self.input_layer = InputLayer(
            window_len=window_len,
            embed_dim=embed_dim,
            args=args
        )
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            *[
                EncoderBlock(
                    embed_dim=embed_dim,
                    nb_head=nb_head,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim*window_len),
            nn.Linear(in_features=embed_dim*window_len, out_features=self.num_classes),
            nn.Linear(in_features=self.num_classes, out_features=self.num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.cuda.set_device(f"cuda:{self.args.cuda_device}")
        x = x.permute(0,2,1)
        # (B, C, W, H) -> (B, N, D)
        out = self.input_layer(x)
        # (B, N, D) -> (B, N, D)
        out = self.encoder(out)
        # extract only class token
        # (B, N, D) -> (B, D)
        a,b,c = out.shape
        # (B, D) -> (B, M)
        pred = self.fc(out.view(a,b*c))
        return pred
if __name__ == "__main__":
    a = torch.rand(32,128,113).cuda()
    model = VisionTransformer(18, 113, 128, 4, 4, 8, 0.3).cuda()
    print(model(a))
