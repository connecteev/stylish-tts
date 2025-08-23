import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .common import LinearNorm
from .text_encoder import FFN, MultiHeadAttention, sequence_mask
from .common import InstanceNorm1d
from .conv_next import ConvNeXtBlock
from .adawin import AdaWinBlock1d, AdaWinLayer1d


class ProsodyEncoder(nn.Module):
    def __init__(
        self,
        sty_dim,
        d_model,
        nlayers,
        dropout=0.1,
        n_heads=2,
        kernel_size=1,
        norm_window_length=17,
        **kwargs,
    ):
        super().__init__()
        hidden_channels = d_model + sty_dim
        self.n_heads = n_heads
        self.n_layers = nlayers
        self.kernel_size = kernel_size

        self.drop = torch.nn.Dropout(dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        self.proj_layers = torch.nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=dropout
                )
            )
            self.norm_layers_1.append(
                AdaWinLayer1d(
                    channels=hidden_channels,
                    window_length=norm_window_length,
                    style_dim=sty_dim,
                )
            )
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels * 2,
                    kernel_size,
                    p_dropout=dropout,
                )
            )
            self.norm_layers_2.append(
                AdaWinLayer1d(
                    channels=hidden_channels,
                    window_length=norm_window_length,
                    style_dim=sty_dim,
                )
            )
            self.proj_layers.append(nn.Conv1d(hidden_channels, d_model, 1))

    def forward(self, x, style, x_lengths):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = torch.cat([x, style], dim=1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y, style, x_lengths)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y, style, x_lengths)
            x = self.proj_layers[i](x)
            x = torch.cat([x, style], dim=1)
        x = x * x_mask
        return x.transpose(-1, -2)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)
