import torch
from torch.nn import functional as F
from einops import rearrange
from .text_encoder import FFN, MultiHeadAttention, sequence_mask


class StyleTransformer(torch.nn.Module):
    def __init__(
        self,
        channels,
        style_dim,
        nlayers,
        dropout=0.1,
        n_heads=2,
        kernel_size=1,
        **kwargs,
    ):
        super().__init__()
        hidden_channels = channels + style_dim
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
            self.norm_layers_1.append(AdaLayerNorm(style_dim, hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels * 2,
                    kernel_size,
                    p_dropout=dropout,
                )
            )
            self.norm_layers_2.append(AdaLayerNorm(style_dim, hidden_channels))
            self.proj_layers.append(torch.nn.Conv1d(hidden_channels, channels, 1))

    def forward(self, x, x_lengths, style):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = torch.cat([x, style], dim=1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](torch.transpose(x + y, -1, -2), style).transpose(
                -1, -2
            )

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](torch.transpose(x + y, -1, -2), style).transpose(
                -1, -2
            )
            x = self.proj_layers[i](x)
            x = torch.cat([x, style], dim=1)
        x = x * x_mask
        return x.transpose(-1, -2)


class AdaLayerNorm(torch.nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = torch.nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        s = rearrange(s, "b s t -> b t s")
        h = self.fc(s)
        gamma = h[:, :, : self.channels]
        beta = h[:, :, self.channels :]

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x
