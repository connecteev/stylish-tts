import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from einops import rearrange
from .common import InstanceNorm1d
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
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
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


def calculate_mean(x, mask, kernel):
    num = convolution(x, kernel)
    denom = convolution(mask, kernel)
    return num / (denom + 1e-9) * mask


def calculate_var(x, mean, mask, kernel):
    term = torch.square((x - mean) * mask)
    num = convolution(term, kernel)
    denom = convolution(mask, kernel)
    return num / (denom + 1e-9) * mask


def convolution(x, kernel):
    x = F.conv1d(x, kernel.to(x.device), padding=kernel.shape[-1] // 2)
    return x


kernel_cache = {}


def kernel_key(norm_type, channels, window_length):
    return f"{norm_type}-{channels}-{window_length}"


def make_kernel(norm_type, channels, window_length, device):
    key = kernel_key(norm_type, channels, window_length)
    if key in kernel_cache:
        return kernel_cache[key]

    if norm_type == "instance":
        kernel = torch.diag(torch.ones(channels)).unsqueeze(2)
        kernel = torch.broadcast_to(kernel, (channels, channels, window_length))
    elif norm_type == "layer":
        kernel = torch.ones(1, channels, window_length)
    else:
        exit("Unsupported norm_type for AdaWin: {norm_type}")
    kernel = kernel.to(device)
    kernel_cache[key] = kernel
    return kernel


class AdaWinInstance1d(nn.Module):
    def __init__(self, *, channels, window_length, style_dim):
        super().__init__()
        self.channels = channels
        self.window_length = window_length
        self.norm = WindowedNorm1d()
        self.fc = weight_norm(nn.Linear(style_dim, channels * 2))

    def forward(self, x, s, lengths):
        s = rearrange(s, "b s t -> b t s")
        h = self.fc(s)
        h = rearrange(h, "b t s -> b s t")

        kernel = make_kernel("instance", self.channels, self.window_length, x.device)
        mask = sequence_mask(lengths, x.shape[2]).unsqueeze(1).to(x.dtype).to(x.device)
        mask = torch.broadcast_to(mask, (x.shape[0], kernel.shape[1], x.shape[2]))
        gamma = h[:, : self.channels, :]
        gamma = calculate_mean(gamma, mask, kernel)
        beta = h[:, self.channels :, :]
        beta = calculate_mean(beta, mask, kernel)
        return (1 + gamma) * self.norm(x, mask, kernel) + beta


class AdaWinLayer1d(nn.Module):
    def __init__(self, *, channels, window_length, style_dim):
        super().__init__()
        self.channels = channels
        self.window_length = window_length
        self.norm = WindowedNorm1d()
        self.fc = nn.Linear(style_dim, 2)

    def forward(self, x, s, lengths):
        s = rearrange(s, "b s t -> b t s")
        h = self.fc(s)
        h = rearrange(h, "b t s -> b s t")

        gb_kernel = make_kernel("layer", 1, self.window_length, x.device)
        mask = sequence_mask(lengths, x.shape[2]).unsqueeze(1).to(x.dtype).to(x.device)
        gamma = h[:, :1, :]
        # gamma = calculate_mean(gamma, mask, gb_kernel)
        beta = h[:, 1:, :]
        # beta = calculate_mean(beta, mask, gb_kernel)
        kernel = make_kernel("layer", self.channels, self.window_length, x.device)
        mask = torch.broadcast_to(mask, (x.shape[0], kernel.shape[1], x.shape[2]))
        return (1 + gamma) * self.norm(x, mask, kernel) + beta


class WindowedNorm1d(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, mask, kernel):
        # x shape: (N, C, L)
        # mean = calculate_mean(x, mask, kernel)
        # var = calculate_var(x, mean, mask, kernel)
        # x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        x_normalized = torch.tanh(self.alpha * x)
        return x_normalized
