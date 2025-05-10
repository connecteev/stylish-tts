import math
import random
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from ..conv_next import ConvNeXtBlock, BasicConvNeXtBlock


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


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        else:
            return torch.nn.functional.interpolate(
                x, scale_factor=2, mode="nearest-exact"
            )


class MelDecoder(nn.Module):
    def __init__(
        self,
        dim_in=512,
        style_dim=128,
        dim_out=512,
        intermediate_dim=1536,
    ):
        super().__init__()

        # TODO Use arguments instead of hardcoding
        self.decode = nn.ModuleList()

        self.encode = AdainResBlk1d(dim_in + 2, dim_in, style_dim)

        self.decode.append(AdainResBlk1d(dim_in + 2 + 64, dim_in, style_dim))
        self.decode.append(AdainResBlk1d(dim_in + 2 + 64, dim_in, style_dim))
        self.decode.append(AdainResBlk1d(dim_in + 2 + 64, dim_in, style_dim))
        self.decode.append(
            AdainResBlk1d(dim_in + 2 + 64, dim_out, style_dim, upsample=True)
        )

        self.F0_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        )

        self.N_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        )

        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(dim_in, 64, kernel_size=1)),
        )

    def forward(self, asr, F0_curve, N, s, probing=False):
        if self.training:
            downlist = [0, 3, 7]
            F0_down = downlist[random.randint(0, 2)]
            downlist = [0, 3, 7, 15]
            N_down = downlist[random.randint(0, 3)]
            if F0_down:
                F0_curve = (
                    nn.functional.conv1d(
                        F0_curve.unsqueeze(1),
                        torch.ones(1, 1, F0_down).to("cuda"),
                        padding=F0_down // 2,
                    ).squeeze(1)
                    / F0_down
                )
            if N_down:
                N = (
                    nn.functional.conv1d(
                        N.unsqueeze(1),
                        torch.ones(1, 1, N_down).to("cuda"),
                        padding=N_down // 2,
                    ).squeeze(1)
                    / N_down
                )

        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))

        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)

        asr_res = self.asr_res(asr)

        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False

        return x, F0_curve
