import torch
from .common import LinearNorm
from .text_encoder import TextEncoder
from .text_style_encoder import TextStyleEncoder
from .prosody_encoder import ProsodyEncoder


class DurationPredictor(torch.nn.Module):
    def __init__(
        self, style_dim, inter_dim, text_config, style_config, duration_config
    ):
        super().__init__()
        self.text_encoder = TextEncoder(inter_dim=inter_dim, config=text_config)
        self.style_encoder = TextStyleEncoder(
            inter_dim,
            style_dim,
            config=style_config,
        )
        self.prosody_encoder = ProsodyEncoder(
            sty_dim=style_dim,
            d_model=inter_dim,
            nlayers=duration_config.n_layer,
            dropout=duration_config.dropout,
        )
        self.dropout = torch.nn.Dropout(duration_config.last_dropout)
        self.duration_proj = LinearNorm(
            inter_dim + style_dim, duration_config.duration_classes
        )

    def forward(self, texts, text_lengths):
        encoding, _, _ = self.text_encoder(texts, text_lengths)
        style = self.style_encoder(encoding, text_lengths)
        prosody = self.prosody_encoder(encoding, style, text_lengths)
        prosody = self.dropout(prosody)
        duration = self.duration_proj(prosody)
        return duration
