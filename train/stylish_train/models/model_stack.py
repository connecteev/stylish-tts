import torch
from .text_encoder import TextEncoder
from .style_encoder import StyleEncoder
from .decoder.mel_decoder import MelDecoder
from .conv_next import ConvNeXtBlock


class ModelStack(torch.nn.Module):
    def __init__(self, model_config):
        super(ModelStack, self).__init__()

        self.num_layers = 8
        self.dim = model_config.n_fft // 2 + 1

        self.text_encoder = TextEncoder(
            channels=self.dim,
            kernel_size=model_config.text_encoder.kernel_size,
            depth=model_config.text_encoder.n_layer,
            n_symbols=model_config.text_encoder.n_token,
        )

        self.style_encoder = StyleEncoder(
            dim_in=model_config.style_encoder.dim_in,
            style_dim=model_config.style_dim,
            max_conv_dim=model_config.style_encoder.hidden_dim,
            skip_downsamples=model_config.style_encoder.skip_downsamples,
        )

        self.decoder = MelDecoder(
            dim_in=self.dim,
            style_dim=model_config.style_dim,
            dim_out=self.dim,
        )

        self.prenorm = torch.nn.LayerNorm(self.dim, eps=1e-6)
        self.convnext = torch.nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=self.dim,
                    dim_out=self.dim,
                    intermediate_dim=self.dim * 2,
                    style_dim=model_config.style_dim,
                    dilation=[1, 3, 5],
                )
                for _ in range(self.num_layers)
            ]
        )
        self.postnorm = torch.nn.LayerNorm(self.dim, eps=1e-6)

    def forward(self, *, text, text_length, duration, mel, pitch, energy, text_mask):
        encoding = self.text_encoder(text, text_length, text_mask)
        style = self.style_encoder(mel.unsqueeze(1))
        decoding, _ = self.decoder(encoding @ duration, pitch, energy, style)
        x = decoding.transpose(1, 2)
        x = self.prenorm(x)
        x = x.transpose(1, 2)
        for block in self.convnext:
            x = block(x, style)
        x = x.transpose(1, 2)
        x = self.postnorm(x)
        return x
