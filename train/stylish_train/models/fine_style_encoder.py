import torch
from torch.nn import functional as F

from .conv_next import BasicConvNeXtBlock

from .conformer import Conformer

# from torchaudio.models import Conformer
from einops import rearrange


class FineStyleEncoder(torch.nn.Module):
    # TODO: Remvoe hard-coded values
    def __init__(self, inter_dim, style_dim, layers):
        super().__init__()
        self.conv_in = torch.nn.Conv1d(inter_dim, 128, kernel_size=7, padding=3)
        # self.conformer = Conformer(
        #     input_dim=512,
        #     num_heads=8,
        #     ffn_dim=inter_dim * 2,
        #     num_layers=layers,
        #     depthwise_conv_kernel_size=7,
        #     dropout=0.1,
        #     use_group_norm=True,
        #     convolution_first=False,
        # )
        self.conformer = Conformer(
            dim=128,
            depth=layers,
            dim_head=32,
            heads=8,
            ff_mult=2,
            conv_expansion_factor=2,
            conv_kernel_size=7,
            attn_dropout=0.3,
            ff_dropout=0.3,
            conv_dropout=0.3,
            use_sdpa=True,
        )
        self.proj_out = torch.nn.Linear(128, style_dim)
        # self.blocks = torch.nn.ModuleList(
        #     [
        #         BasicConvNeXtBlock(
        #             dim=style_dim,
        #             intermediate_dim=style_dim * 4,
        #         )
        #         for _ in range(layers)
        #     ]
        # )

    def forward(self, x, lengths):
        x = self.conv_in(x)
        # for block in self.blocks:
        #     x = block(x)
        # return x
        x = rearrange(x, "b c l -> b l c")
        x = self.conformer(x, lengths)
        x = self.proj_out(x)
        x = rearrange(x, "b l c -> b c l")
        # x = (
        #     F.conv2d(
        #         x.unsqueeze(1),
        #         torch.ones(1, 1, 1, 37).to(x.device),
        #         padding=(0, 37 // 2),
        #         stride=1,
        #     ).squeeze(1)
        #     / 37
        # )
        return x
