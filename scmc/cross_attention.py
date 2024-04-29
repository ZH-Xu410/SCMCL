import math
import warnings

import torch
from torch import nn


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 q_dim,
                 k_dim,
                 v_dim,
                 heads=8,
                 dropout=0.1,
                 attention_dropout=0.1,
                 qkv_bias=True):
        super(MultiHeadAttention, self).__init__()
        assert all([dim % heads == 0 for dim in (q_dim, k_dim, v_dim)])
        self.heads = heads
        self.scale = (q_dim // heads) ** -0.5

        self.to_q = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.to_k = nn.Linear(k_dim, q_dim, bias=qkv_bias)
        self.to_v = nn.Linear(v_dim, q_dim, bias=qkv_bias)
        self.to_out = nn.Linear(q_dim, q_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v):
        B, Nq, C = q.shape
        Nk = k.shape[1]
        Nv = v.shape[1]
        assert Nk == Nv

        q = self.to_q(q).reshape(B, Nq, self.heads, -1).transpose(1, 2)
        k = self.to_k(k).reshape(B, Nk, self.heads, -1).transpose(1, 2)
        v = self.to_v(v).reshape(B, Nv, self.heads, -1).transpose(1, 2)
        
        # B H Nq C @ B H C Nk -> B H Nq Nk
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        # B H Nq Nk @ B H Nv C -> B H Nq C; B Nq H C
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.to_out(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.1, activation=nn.GELU):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():

        v = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # [2v-1, 2u-1].
        tensor.uniform_(2 * v - 1, 2 * u - 1)

        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, heads, dropout=0, attention_dropout=0, drop_path=0, 
                 qkv_bias=True, normalize=nn.LayerNorm, activation=nn.GELU):
        super().__init__()
        self.norm_q = normalize(q_dim)
        self.norm_k = normalize(k_dim)
        self.norm_v = normalize(v_dim)
        self.attention = MultiHeadAttention(q_dim, k_dim, v_dim, heads, dropout, attention_dropout, qkv_bias)
        self.norm = normalize(q_dim)
        self.ffn = FeedForward(q_dim, q_dim*2, dropout, activation=activation)
        self.drop_path = DropPath(drop_path)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, q, k, v):
        x = self.attention(
            self.norm_q(q), 
            self.norm_k(k), 
            self.norm_v(v))
        x = self.attention(q, k, v)
        x = self.drop_path(x) + q
        y = self.ffn(self.norm(x))
        y = self.drop_path(y) + x

        return y


class CrossAttentionSeq(nn.Module):
    def __init__(self, in_features, dim, num_blocks, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.proj_q = nn.Linear(in_features, dim)
        self.proj_x = nn.Linear(in_features, dim)
        self.blocks = nn.ModuleList([CrossAttentionBlock(dim, dim, dim, *args, **kwargs) for _ in range(num_blocks)])
    
    def forward(self, q, x):
        input_4D = False
        if len(q.shape) == 4:
            H, W = q.shape[2:]
            q = q.flatten(2).transpose(2, 1)
            x = x.flatten(2).transpose(2, 1)
            input_4D = True
        else:
            q = q.transpose(2, 1)
            x = x.transpose(2, 1)

        if q.shape[-1] != self.dim:
            q = self.proj_q(q)
        if x.shape[-1] != self.dim:
            x = self.proj_x(x)
        for m in self.blocks:
            x = m(q, x, x)

        if input_4D:
            B, _, C = x.shape
            x = x.transpose(2, 1).view(B, C, H, W)
        else:
            x = x.transpose(2, 1)

        return x


if __name__ == '__main__':
    x = torch.randn(1, 128, 5)
    y = torch.randn(1, 128, 5)
    cross_attn = CrossAttentionSeq(2, 128, 128, 128, 4)
    z = cross_attn(x, y)
    print(z.shape, z.mean())
