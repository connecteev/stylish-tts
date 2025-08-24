import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from einops import rearrange
from .common import InstanceNorm1d
from .adawin import AdaWinBlock1d, AdaPitchBlock1d
from .text_encoder import TextEncoder, MultiHeadAttention
from .fine_style_encoder import FineStyleEncoder
from .prosody_encoder import ProsodyEncoder
from utils import length_to_mask


class PitchEnergyPredictor(torch.nn.Module):
    def __init__(
        self,
        style_dim,
        inter_dim,
        text_config,
        style_config,
        duration_config,
        pitch_energy_config,
    ):
        super().__init__()
        # self.speaker_encoder = nn.Embedding(30, style_dim)
        # self.text_encoder = TextEncoder(inter_dim=512, config=text_config)
        # self.style_encoder = FineStyleEncoder(inter_dim, style_dim, config=style_config)
        # self.prosody_encoder = DurationEncoder(
        #     sty_dim=style_dim,
        #     d_model=512,
        #     nlayers=3,
        #     dropout=0.2,
        # )
        self.prosody_encoder = ProsodyEncoder(
            sty_dim=style_dim,
            d_model=512,
            nlayers=3,
            dropout=0.2,
        )

        d_hid = inter_dim
        dropout = pitch_energy_config.dropout

        cross_channels = 512 + style_dim
        self.query_norm = AdaLayerNorm(style_dim, cross_channels)
        self.key_norm = AdaLayerNorm(style_dim, cross_channels)
        self.cross_attention = MultiHeadAttention(
            channels=cross_channels,
            out_channels=cross_channels,
            n_heads=8,
            p_dropout=dropout,
        )
        self.cross_window = 5

        self.cross_post = nn.Sequential(
            weight_norm(
                nn.Conv1d(
                    cross_channels,
                    cross_channels,
                    kernel_size=5,
                    padding=2,
                    groups=cross_channels,
                )
            ),
            nn.SiLU(),
            weight_norm(nn.Conv1d(cross_channels, cross_channels, kernel_size=1)),
        )
        # self.shared = nn.LSTM(
        #     512 + style_dim, 512 // 2, 1, batch_first=True, bidirectional=True
        # )

        # self.F0_in = nn.Conv1d(d_hid + 1, 512, 1, 1, 0)
        self.F0 = nn.ModuleList(
            [
                # AdaWinBlock1d(
                #     dim_in=d_hid,
                #     dim_out=d_hid,
                #     style_dim=style_dim,
                #     window_length=norm_window_length,
                #     dropout_p=dropout,
                # )
                AdainResBlk1d(
                    512 + style_dim, 512 + style_dim, style_dim, dropout_p=dropout
                )
                # AdaPitchBlock1d(
                #     channels=128,
                #     style_dim=style_dim,
                #     window_length=norm_window_length,
                #     kernel_size=23,
                #     dilation=[1, 3, 1],
                #     dropout=dropout,
                # )
                # for _ in range(4)
                for _ in range(3)
            ]
        )

        self.N = nn.ModuleList(
            [
                AdainResBlk1d(
                    512 + style_dim, 512 + style_dim, style_dim, dropout_p=dropout
                )
                # AdaWinBlock1d(
                #     dim_in=d_hid,
                #     dim_out=d_hid,
                #     style_dim=style_dim,
                #     window_length=norm_window_length,
                #     dropout_p=dropout,
                # )
                for _ in range(3)
            ]
        )

        self.F0_proj = nn.Conv1d(512 + style_dim, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(512 + style_dim, 1, 1, 1, 0)

    def compute_cross(self, text_encoding, alignment, style, text_mask):
        """
        d_tok: [B, T, C] token states (style-conditioned)
        alignment: [B, T, F] monotonic alignment
        style: [B, S]
        text_mask: [B, T] True at padding
        Returns: en [B, C, F]
        """
        base = torch.matmul(text_encoding.transpose(1, 2), alignment)
        # Build queries from the same base (frame-level)
        query = base.transpose(1, 2)
        query = self.query_norm(query, style).transpose(1, 2)
        key = text_encoding  # [B, T, C]
        key = self.key_norm(key, style).transpose(1, 2)

        # Monotonic band mask
        attention_mask = build_monotonic_band_mask(
            alignment, text_mask, self.cross_window
        )  # [B, 1, F, T]
        # Optional: simple ALiBi-like negative bias to prefer nearby tokens
        # rel_bias = None
        # if self.use_rel_bias:
        #     with torch.no_grad():uery
        #         B, T, F = alignment.shape
        #         tau = alignment.argmax(dim=1)  # [B, F]
        #         t_idx = torch.arange(
        #             T, device=alignment.device
        #         ).view(1, 1, T)
        #         dist = (t_idx - tau.unsqueeze(-1)).abs().float()
        #         # slope tuned small to avoid over-biasing
        #         rel_bias = (-0.05 * dist).unsqueeze(1)  # [B, 1, F, T]
        attention = self.cross_attention(
            query, key, attn_mask=attention_mask
        )  # , rel_bias=rel_bias)
        # attention = attention.transpose(1, 2)
        attention = self.cross_post(attention)
        return (base + attention) / math.sqrt(2.0)

    def forward(self, text_encoding, text_lengths, alignment, style):
        mask = length_to_mask(text_lengths, text_encoding.shape[2]).to(
            text_lengths.device
        )
        prosody = self.prosody_encoder(text_encoding, style, text_lengths)
        x = self.compute_cross(prosody, alignment, style, mask.squeeze(1))

        # x = prosody
        # x = rearrange(x, "b l c -> b c l")
        # x = x @ alignment
        # x = rearrange(x, "b c l -> b l c")
        # x, _ = self.shared(x)

        F0 = x  # .transpose(1, 2)
        for block in self.F0:
            F0 = block(F0, style)
        F0 = self.F0_proj(F0)

        N = x  # .transpose(1, 2)
        for block in self.N:
            N = block(N, style)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)


def build_monotonic_band_mask(alignment, text_mask, window):
    """
    alignment: [B, T, F] (monotonic hard/soft align)
    text_mask: [B, T] True at padding
    Returns attn_mask: [B, 1, F, T] True where attention is NOT allowed.
    """
    with torch.no_grad():
        B, T, F = alignment.shape
        tau = alignment.argmax(dim=1)
        t_idx = torch.arange(T, device=alignment.device).view(1, 1, T)
        tau_exp = tau.unsqueeze(-1)
        band = (t_idx >= (tau_exp - window)) & (t_idx <= (tau_exp + window))

        band_mask = ~band

        # Also mask padded tokens
        key_pad = text_mask.unsqueeze(1).expand(B, F, T)
        full_mask = band_mask | key_pad
        return full_mask.unsqueeze(1)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0,
                )
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], dim=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu()
        x = x.transpose(1, 2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(1, 2), style).transpose(1, 2)
                x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(1, 2), 0.0)
            else:
                x = x.transpose(1, 2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False
                )
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(1, 2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad.to(x.device)
        return x.transpose(1, 2)

    def infer(self, x, style):
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], dim=-1)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(1, 2), style).transpose(1, 2)
                x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
            else:
                x = x.transpose(1, 2)
                x, _ = block(x)
                x = x.transpose(1, 2)
        return x.transpose(1, 2)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(1, 2)
        x = x.transpose(1, 2)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, 2), beta.transpose(1, 2)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, 2).transpose(1, 2)


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        style_dim=64,
        actv=nn.LeakyReLU(0.2),
        upsample="none",
        dropout_p=0.0,
    ):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    groups=dim_in,
                    padding=1,
                    output_padding=1,
                )
            )

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
