import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from einops import rearrange
from .common import InstanceNorm1d, get_padding, init_weights
from .text_encoder import sequence_mask


class AdaWinBlock1d(nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        dim_out,
        style_dim=64,
        actv=nn.LeakyReLU(0.2),
        window_length=37,
        dropout_p=0.0,
    ):
        super().__init__()
        self.actv = actv
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim, window_length)
        self.dropout = nn.Dropout(dropout_p)

    def _build_weights(self, dim_in, dim_out, style_dim, window_length):
        self.conv1 = weight_norm(
            nn.Conv1d(dim_in, dim_out, 3, 1, 1, padding_mode="reflect")
        )
        self.conv2 = weight_norm(
            nn.Conv1d(dim_out, dim_out, 3, 1, 1, padding_mode="reflect")
        )
        self.norm1 = AdaWinInstance1d(
            channels=dim_in, window_length=window_length, style_dim=style_dim
        )
        self.norm2 = AdaWinInstance1d(
            channels=dim_out, window_length=window_length, style_dim=style_dim
        )
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, lengths):
        x = self.norm1(x, s, lengths)
        x = self.actv(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s, lengths)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s, lengths):
        out = self._residual(x, s, lengths)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class AdaConvBlock1d(torch.nn.Module):
    def __init__(
        self, *, channels, style_dim, window_length, kernel_size, dilation, dropout
    ):
        super(AdaConvBlock1d, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                        padding_mode="reflect",
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                        padding_mode="reflect",
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                        padding_mode="reflect",
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                        padding_mode="reflect",
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                        padding_mode="reflect",
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                        padding_mode="reflect",
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

        self.adain1 = nn.ModuleList(
            [
                AdaWinInstance1d(
                    channels=channels, window_length=window_length, style_dim=style_dim
                ),
                AdaWinInstance1d(
                    channels=channels, window_length=window_length, style_dim=style_dim
                ),
                AdaWinInstance1d(
                    channels=channels, window_length=window_length, style_dim=style_dim
                ),
            ]
        )

        self.adain2 = nn.ModuleList(
            [
                AdaWinInstance1d(
                    channels=channels, window_length=window_length, style_dim=style_dim
                ),
                AdaWinInstance1d(
                    channels=channels, window_length=window_length, style_dim=style_dim
                ),
                AdaWinInstance1d(
                    channels=channels, window_length=window_length, style_dim=style_dim
                ),
            ]
        )

        self.alpha1 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))]
        )
        self.alpha2 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, s, lengths):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            xt = n1(x, s, lengths)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(self.dropout(xt))
            xt = n2(xt, s, lengths)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(self.dropout(xt))
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AdaPitchBlock1d(torch.nn.Module):
    def __init__(
        self, *, channels, style_dim, window_length, kernel_size, dilation, dropout
    ):
        super(AdaPitchBlock1d, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[i],
                        padding=get_padding(kernel_size, dilation[i]),
                        padding_mode="reflect",
                    )
                )
                for i in range(3)
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                        padding_mode="reflect",
                    )
                )
                for _ in range(3)
            ]
        )
        self.convs2.apply(init_weights)

        self.adain1 = nn.ModuleList(
            [
                AdaWinInstance1d(
                    channels=channels, window_length=window_length, style_dim=style_dim
                )
                for _ in range(3)
            ]
        )

        self.adain2 = nn.ModuleList(
            [
                AdaWinInstance1d(
                    channels=channels, window_length=window_length, style_dim=style_dim
                ),
                AdaWinInstance1d(
                    channels=channels, window_length=window_length, style_dim=style_dim
                ),
                AdaWinInstance1d(
                    channels=channels, window_length=window_length, style_dim=style_dim
                ),
            ]
        )

        self.alpha1 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))]
        )
        self.alpha2 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, s, lengths):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            xt = n1(x, s, lengths)
            xt = c1(self.dropout(xt))
            xt = n2(xt, s, lengths)
            xt = c2(self.dropout(xt))
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


def calculate_mean(x, mask, kernel, window_length):
    num = convolution(x, kernel, window_length)
    denom = convolution(mask, kernel, window_length)
    return num / (denom + 1e-9) * mask


def calculate_var(x, mean, mask, kernel, window_length):
    term = torch.square((x - mean) * mask)
    num = convolution(term, kernel, window_length)
    denom = convolution(mask, kernel, window_length)
    return num / (denom + 1e-9) * mask


def convolution(x, kernel, window_length):
    x = F.conv1d(x, kernel.to(x.device), padding=window_length // 2)
    return x


kernel_cache = {}


def kernel_key(norm_type, channels, window_length):
    return f"{norm_type}-{channels}-{window_length}"


def make_kernel(norm_type, channels, window_length, device):
    # key = kernel_key(norm_type, channels, window_length)
    # if key in kernel_cache:
    #     return kernel_cache[key]

    if norm_type == "instance":
        kernel = torch.diag(torch.ones(channels)).unsqueeze(2)
        kernel = torch.broadcast_to(kernel, (channels, channels, window_length))
    elif norm_type == "layer":
        kernel = torch.ones(1, channels, window_length)
    else:
        exit("Unsupported norm_type for AdaWin: {norm_type}")
    kernel = kernel.to(device)
    # kernel_cache[key] = kernel
    return kernel


def make_kernel_instance(channels, window_length, device):
    key = kernel_key("instance", channels, window_length)
    if key in kernel_cache:
        return kernel_cache[key]
    kernel = torch.diag(torch.ones(channels)).unsqueeze(2)
    kernel = torch.broadcast_to(kernel, (channels, channels, window_length))
    kernel = kernel.to(device)
    kernel_cache[key] = kernel
    return kernel


def make_kernel_layer(channels, window_length, device):
    key = kernel_key("layer", channels, window_length)
    if key in kernel_cache:
        return kernel_cache[key]
    kernel = torch.ones(1, channels, window_length)
    kernel = kernel.to(device)
    kernel_cache[key] = kernel
    return kernel


class AdaWinInstance1d(nn.Module):
    def __init__(self, *, channels, window_length, style_dim):
        super().__init__()
        self.channels = channels
        self.window_length = window_length
        self.norm = WindowedNorm1d(channels=channels)
        self.fc = weight_norm(nn.Linear(style_dim, channels * 2))
        self.pool = nn.AvgPool1d(
            kernel_size=window_length,
            stride=1,
            padding=window_length // 2,
            count_include_pad=False,
        )

    def forward(self, x, s, lengths):
        s = rearrange(s, "b s t -> b t s")
        h = self.fc(s)
        h = rearrange(h, "b t s -> b s t")

        # kernel = make_kernel_instance(self.channels, self.window_length, x.device)
        # mask = sequence_mask(lengths, x.shape[2]).unsqueeze(1).to(x.dtype).to(x.device)
        # mask = torch.broadcast_to(mask, (x.shape[0], kernel.shape[1], x.shape[2]))
        gamma = h[:, : self.channels, :]
        gamma = self.pool(gamma)
        # gamma = calculate_mean(gamma, mask, kernel, self.window_length)
        beta = h[:, self.channels :, :]
        beta = self.pool(beta)
        # beta = calculate_mean(beta, mask, kernel, self.window_length)
        return (1 + gamma) * self.norm(x) + beta


class AdaWinLayer1d(nn.Module):
    def __init__(self, *, channels, window_length, style_dim):
        super().__init__()
        self.channels = channels
        self.window_length = window_length
        self.norm = WindowedNorm1d(channels=1)
        self.fc = nn.Linear(style_dim, 2)
        self.pool = nn.AvgPool1d(
            kernel_size=window_length,
            stride=1,
            padding=window_length // 2,
            count_include_pad=False,
        )

    def forward(self, x, s, lengths):
        s = rearrange(s, "b s t -> b t s")
        h = self.fc(s)
        h = rearrange(h, "b t s -> b s t")

        # gb_kernel = make_kernel_layer(1, self.window_length, x.device)
        # mask = sequence_mask(lengths, x.shape[2]).unsqueeze(1).to(x.dtype).to(x.device)
        gamma = h[:, :1, :]
        gamma = self.pool(gamma)
        # gamma = calculate_mean(gamma, mask, gb_kernel, self.window_length)
        beta = h[:, 1:, :]
        beta = self.pool(beta)
        # beta = calculate_mean(beta, mask, gb_kernel, self.window_length)
        # kernel = make_kernel_layer(self.channels, self.window_length, x.device)
        # mask = torch.broadcast_to(mask, (x.shape[0], kernel.shape[1], x.shape[2]))
        return (1 + gamma) * self.norm(x) + beta


class WindowedNorm1d(torch.nn.Module):
    def __init__(self, *, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(channels) * 0.5)

    def forward(self, x):
        # x shape: (N, C, L)
        # mean = calculate_mean(x, mask, kernel)
        # var = calculate_var(x, mean, mask, kernel)
        # x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        alpha = rearrange(self.alpha, "a -> 1 a 1")
        x_normalized = torch.tanh(alpha * x)
        return x_normalized
