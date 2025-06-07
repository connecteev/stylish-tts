import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .text_encoder import TextEncoder
from .fine_style_encoder import FineStyleEncoder
from .style_transformer import StyleTransformer
from .conv_next import ConvNeXtBlock


class DurationPredictor(nn.Module):
    def __init__(
        self,
        *,
        style_dim,
        d_hid,
        nlayers,
        max_dur,
        dropout,
        text_encoder_config,
        style_layers,
        conv_layers
    ):
        super().__init__()
        self.text_encoder = TextEncoder(inter_dim=d_hid, config=text_encoder_config)
        self.style_encoder = FineStyleEncoder(
            inter_dim=d_hid, style_dim=style_dim, layers=style_layers
        )
        # TODO Fix hardcoded values
        self.transformer = StyleTransformer(
            channels=d_hid,
            style_dim=style_dim,
            nlayers=nlayers,
            dropout=dropout,
            n_heads=2,
            kernel_size=1,
        )
        hidden_dim = d_hid + style_dim
        self.conv = torch.nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=hidden_dim,
                    dim_out=hidden_dim,
                    intermediate_dim=d_hid * 4,
                    style_dim=style_dim,
                    dilation=[1, 3, 5],
                    activation=True,
                    dropout=dropout,
                )
                for _ in range(conv_layers - 1)
            ]
        )
        self.conv.append(
            ConvNeXtBlock(
                dim_in=hidden_dim,
                dim_out=1,
                intermediate_dim=d_hid * 4,
                style_dim=style_dim,
                dilation=[1, 3, 5],
                activation=True,
                dropout=dropout,
            )
        )

    def forward(self, texts, text_lengths, alignment=None):
        if alignment is not None:
            texts = rearrange(texts, "b k -> b 1 k")
            texts = texts.to(torch.float) @ alignment
            texts = torch.repeat_interleave(texts, repeats=2, dim=2)
            texts = rearrange(texts, "b 1 t -> b t").to(torch.long)
        text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        style = self.style_encoder(text_encoding)
        x = self.transformer(text_encoding, text_lengths, style)
        x = rearrange(x, "b t c -> b c t")
        for block in self.conv:
            x = block(x, style)
        return rearrange(x, "b 1 t -> b t")
