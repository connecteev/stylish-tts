import torch
from torch.nn import functional as F

# from .conformer import Conformer
from torchaudio.models import Conformer
from einops import rearrange


class FineStyleEncoder(torch.nn.Module):
    # TODO: Remvoe hard-coded values
    def __init__(self, inter_dim, style_dim, layers):
        super().__init__()
        self.conformer = Conformer(
            input_dim=inter_dim,
            num_heads=8,
            ffn_dim=inter_dim * 2,
            num_layers=layers,
            depthwise_conv_kernel_size=7,
            dropout=0.3,
            use_group_norm=False,
            convolution_first=False,
        )
        # self.conformer = Conformer(
        #     dim=inter_dim,
        #     depth=layers,
        #     dim_head=64,
        #     heads=8,
        #     ff_mult=4,
        #     conv_expansion_factor=2,
        #     conv_kernel_size=31,
        #     attn_dropout=0.3,
        #     ff_dropout=0.3,
        #     conv_dropout=0.3,
        # )
        self.conv_out = torch.nn.Conv1d(inter_dim, style_dim, kernel_size=7, padding=3)

    def forward(self, x):
        lengths = torch.full(size=(x.shape[0],), fill_value=x.shape[2], device=x.device)
        x = rearrange(x, "b c l -> b l c")
        x, _ = self.conformer(x, lengths)
        x = rearrange(x, "b l c -> b c l")
        x = self.conv_out(x)
        x = (
            F.conv2d(
                x.unsqueeze(1),
                torch.ones(1, 1, 1, 37).to(x.device),
                padding=(0, 37 // 2),
                stride=1,
            ).squeeze(1)
            / 37
        )
        return x
