import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from einops import rearrange
from .common import InstanceNorm1d
from .adawin import AdaWinBlock1d


class PitchEnergyPredictor(torch.nn.Module):
    def __init__(self, style_dim, d_hid, dropout=0.1):
        super().__init__()
        norm_window_length = 17
        self.shared = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )
        self.F0 = nn.ModuleList()
        self.F0.append(
            AdaWinBlock1d(
                dim_in=d_hid,
                dim_out=d_hid,
                style_dim=style_dim,
                window_length=norm_window_length,
                dropout_p=dropout,
            )
        )
        self.F0.append(
            AdaWinBlock1d(
                dim_in=d_hid,
                dim_out=d_hid // 2,
                style_dim=style_dim,
                window_length=norm_window_length,
                dropout_p=dropout,
            )
        )
        self.F0.append(
            AdaWinBlock1d(
                dim_in=d_hid // 2,
                dim_out=d_hid // 2,
                style_dim=style_dim,
                window_length=norm_window_length,
                dropout_p=dropout,
            )
        )

        self.N = nn.ModuleList()
        self.N.append(
            AdaWinBlock1d(
                dim_in=d_hid,
                dim_out=d_hid,
                style_dim=style_dim,
                window_length=norm_window_length,
                dropout_p=dropout,
            )
        )
        self.N.append(
            AdaWinBlock1d(
                dim_in=d_hid,
                dim_out=d_hid // 2,
                style_dim=style_dim,
                window_length=norm_window_length,
                dropout_p=dropout,
            )
        )
        self.N.append(
            AdaWinBlock1d(
                dim_in=d_hid // 2,
                dim_out=d_hid // 2,
                style_dim=style_dim,
                window_length=norm_window_length,
                dropout_p=dropout,
            )
        )

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, prosody, style, lengths):
        # x = torch.cat([prosody, style], dim=1)
        x = prosody
        x, _ = self.shared(x.transpose(-1, -2))

        s = style
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s, lengths)
        F0 = self.F0_proj(F0)

        s = style
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s, lengths)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)
