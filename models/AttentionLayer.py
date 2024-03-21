import torch.nn as nn
from .transformer.layers import MultiHeadAttention, LayerNorm
from .transformer.layers import PositionwiseFeedForward


class SelfAttentionLayer(nn.Module):
    def __init__(self, *, input_dim, d_model, n_head, drop_prob=0.1, residual=True):
        super(SelfAttentionLayer, self).__init__()
        self.atten = MultiHeadAttention(q_dim=input_dim, k_dim=input_dim, d_model=d_model, n_head=n_head)
        self.norm = LayerNorm(d_model=input_dim)
        self.dropout = nn.Dropout(p=drop_prob)
        self.residual = residual

    def forward(self, x):
        """
        :param x: (bs, n_img, feature_dim)
        :return: enhanced x:(bs, n_img, feature_dim)
        """
        _x = x
        x = self.atten(x, x, x)
        x = self.dropout(x)
        if self.residual:
            out = self.norm(x + _x)
        else:
            out = self.norm(x)
        return out


class CrossAttentionLayer(nn.Module):
    def __init__(self, *, q_dim, k_dim, d_model, n_head, drop_prob=0.1, residual=True):
        super(CrossAttentionLayer, self).__init__()
        self.atten = MultiHeadAttention(q_dim=q_dim, k_dim=k_dim, d_model=d_model, n_head=n_head)
        self.norm = LayerNorm(d_model=q_dim)
        self.dropout = nn.Dropout(p=drop_prob)
        self.residual = residual


    def forward(self, q, k, v, mask=None):
        """
        :param q, k, v: (bs, n_img, feature_dim)
        :param mask: (bs, n_img, H*W)
        :return: enhanced q:(bs, n_img, feature_dim)
        """
        _x = q
        x = self.atten(q, k, v, mask)
        x = self.dropout(x)
        if self.residual:
            out = self.norm(x + _x)
        else:
            out = self.norm(x)
        return out

class CrossAttentionFFNLayer(nn.Module):
    def __init__(self, *, q_dim, k_dim, d_model, n_head, drop_prob=0.1, residual=True):
        super(CrossAttentionLayer, self).__init__()
        self.atten = MultiHeadAttention(q_dim=q_dim, k_dim=k_dim, d_model=d_model, n_head=n_head)
        self.norm = LayerNorm(d_model=q_dim)
        self.dropout = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=d_model//2, drop_prob=drop_prob)
        self.residual = residual

    def forward(self, q, k, v, mask=None):
        """
        :param q, k, v: (bs, n_img, feature_dim)
        :param mask: (bs, n_img, H*W)
        :return: enhanced q:(bs, n_img, feature_dim)
        """
        _x = q
        x = self.atten(q, k, v, mask)
        x = self.dropout(x)
        if self.residual:
            x = self.norm(x + _x)
            _x = x
        else:
            x = self.norm(x)
        x = self.ffn(x)
        if self.residual:
            out = self.norm(x + _x)
        else:
            out = self.norm(x)
        return out


if __name__ == "__main__":
    import torch

    device = torch.device("cuda")
    data = torch.ones((112, 50, 512)).to(device)
    attn = SelfAttentionLayer(d_model=512, n_head=4).to(device)
    y = attn(data)
    print(y)
