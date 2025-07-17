import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from einops import rearrange
from .common import InstanceNorm1d
from .adawin import AdaWinBlock1d, AdaPitchBlock1d
from .text_encoder import TextEncoder
from .fine_style_encoder import FineStyleEncoder
from .prosody_encoder import ProsodyEncoder
from .rmvpe import PitchModel


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
        self.text_encoder = TextEncoder(inter_dim=inter_dim, config=text_config)
        self.style_encoder = FineStyleEncoder(inter_dim, style_dim, config=style_config)
        self.prosody_encoder = ProsodyEncoder(
            sty_dim=style_dim,
            d_model=inter_dim,
            nlayers=duration_config.n_layer,
            dropout=duration_config.dropout,
        )

        d_hid = inter_dim
        dropout = pitch_energy_config.dropout
        norm_window_length = 37
        self.shared = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )

        # self.F0_in = nn.Conv1d(d_hid, 128, 1, 1, 0)
        # self.F0 = nn.ModuleList(
        #     [
        #         # AdaWinBlock1d(
        #         #     dim_in=d_hid,
        #         #     dim_out=d_hid,
        #         #     style_dim=style_dim,
        #         #     window_length=norm_window_length,
        #         #     dropout_p=dropout,
        #         # )
        #         AdaPitchBlock1d(
        #             channels=128,
        #             style_dim=style_dim,
        #             window_length=norm_window_length,
        #             kernel_size=23,
        #             dilation=[1, 3, 1],
        #             dropout=dropout,
        #         )
        #         for _ in range(4)
        #     ]
        # )
        self.pitch_model = PitchModel(
            n_blocks=4,
            n_gru=1,
            kernel_size=(2, 2),
        )

        self.N = nn.ModuleList(
            [
                AdaWinBlock1d(
                    dim_in=d_hid,
                    dim_out=d_hid,
                    style_dim=style_dim,
                    window_length=norm_window_length,
                    dropout_p=dropout,
                )
                for _ in range(3)
            ]
        )

        # self.F0_proj = nn.Conv1d(128, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid, 1, 1, 1, 0)

    def forward(self, texts, lengths, alignment):
        encoding, _, _ = self.text_encoder(texts, lengths)
        style = self.style_encoder(encoding, lengths)
        prosody = self.prosody_encoder(encoding, style, lengths)
        s = style @ alignment

        x = prosody
        x = rearrange(x, "b l c -> b c l")
        x = x @ alignment
        x = rearrange(x, "b c l -> b l c")

        # F0 = x.transpose(-1, -2)
        # F0 = self.F0_in(F0)
        # for block in self.F0:
        #     F0 = block(F0, s, lengths)
        # F0 = self.F0_proj(F0)
        hidden_vec, F0 = self.pitch_model(x)

        x, _ = self.shared(x)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s, lengths)
        N = self.N_proj(N)

        return F0, N.squeeze(1), hidden_vec
