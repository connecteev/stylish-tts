import torch
from .text_encoder import TextEncoder
from .text_style_encoder import FineStyleEncoder
from .prosody_encoder import ProsodyEncoder
from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor
from .decoder import Decoder
from .generator import Generator


class SpeechPredictor(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.text_encoder = TextEncoder(
            inter_dim=model_config.inter_dim, config=model_config.text_encoder
        )

        self.style_encoder = FineStyleEncoder(
            model_config.inter_dim,
            model_config.style_dim,
            model_config.style_encoder,
        )

        self.decoder = Decoder(
            dim_in=model_config.inter_dim,
            style_dim=model_config.style_dim,
            dim_out=model_config.generator.upsample_initial_channel,
            hidden_dim=model_config.decoder.hidden_dim,
            residual_dim=model_config.decoder.residual_dim,
        )

        self.generator = Generator(
            style_dim=model_config.style_dim,
            resblock_kernel_sizes=model_config.generator.resblock_kernel_sizes,
            upsample_rates=model_config.generator.upsample_rates,
            upsample_initial_channel=model_config.generator.upsample_initial_channel,
            resblock_dilation_sizes=model_config.generator.resblock_dilation_sizes,
            upsample_kernel_sizes=model_config.generator.upsample_kernel_sizes,
            gen_istft_n_fft=model_config.generator.gen_istft_n_fft,
            gen_istft_hop_size=model_config.generator.gen_istft_hop_size,
            sample_rate=model_config.sample_rate,
        )

    def forward(self, texts, text_lengths, alignment, pitch, energy):
        text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        style = self.style_encoder(text_encoding, text_lengths)
        mel, f0_curve = self.decoder(
            text_encoding @ alignment,
            pitch,
            energy,
            style @ alignment,
        )
        prediction = self.generator(
            mel=mel,
            style=style @ alignment,
            pitch=f0_curve,
            energy=energy,
        )
        return prediction
