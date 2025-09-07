import torch
from einops import rearrange


class ExportModel(torch.nn.Module):
    def __init__(
        self,
        *,
        speech_predictor,
        duration_predictor,
        pitch_energy_predictor,
        pe_text_encoder,
        pe_text_style_encoder,
        device,
        **kwargs
    ):
        super(ExportModel, self).__init__()

        for model in [
            speech_predictor,
            duration_predictor,
            pitch_energy_predictor,
            pe_text_encoder,
            pe_text_style_encoder,
        ]:
            model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False

        self.speech_predictor = speech_predictor
        self.pitch_energy_predictor = pitch_energy_predictor
        self.pe_text_encoder = pe_text_encoder
        self.pe_text_style_encoder = pe_text_style_encoder

    def forward(self, texts, text_lengths, alignment):
        pe_text_encoding, _, _ = self.pe_text_encoder(texts, text_lengths)
        pe_text_style = self.pe_text_style_encoder(pe_text_encoding, text_lengths)
        pitch, energy = self.pitch_energy_predictor(
            pe_text_encoding, text_lengths, alignment, pe_text_style
        )
        prediction = self.speech_predictor(
            texts, text_lengths, alignment, pitch, energy
        )
        audio = rearrange(prediction.audio, "1 1 l -> l")
        return audio
