import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .common import LinearNorm


class DurationPredictor(nn.Module):
    def __init__(
        self,
        inter_dim,
        style_dim,
        max_dur,
    ):
        super().__init__()
        self.duration_proj = LinearNorm(inter_dim + style_dim, max_dur)

    def forward(self, d):
        duration = self.duration_proj(d)
        return duration.squeeze(-1)
