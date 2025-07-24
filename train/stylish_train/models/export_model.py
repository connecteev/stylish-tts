import torch
from ..util import duration_to_alignment


class ExportModel(torch.nn.Module):
    def __init__(
        self,
        *,
        speech_predictor,
        duration_predictor,
        pitch_energy_predictor,
        device="cuda",
        **kwargs
    ):
        super(ExportModel, self).__init__()

        for model in [
            speech_predictor,
            duration_predictor,
            pitch_energy_predictor,
        ]:
            model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False

        self.device = device
        self.speech_predictor = speech_predictor
        self.duration_predictor = duration_predictor
        self.pitch_energy_predictor = pitch_energy_predictor

    def duration_predict(self, texts, text_lengths):
        duration = self.duration_predictor(texts, text_lengths)
        duration = torch.sigmoid(duration).sum(axis=-1)
        duration = torch.round(duration).clamp(min=1).long().squeeze()

        result = duration_to_alignment(duration)
        result = result.unsqueeze(0)
        return result

    def forward(self, texts, text_lengths):
        alignment = self.duration_predict(texts, text_lengths)
        pitch, energy = self.pitch_energy_predictor(texts, text_lengths, alignment)
        kernel = torch.FloatTensor(
            [[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]
        ).to(texts.device)
        pitch = torch.nn.functional.conv1d(pitch, kernel, padding=3)
        energy = torch.nn.functional.conv1d(energy, kernel, padding=3)
        prediction = self.speech_predictor(
            texts, text_lengths, alignment, pitch, energy
        )
        return prediction.audio.squeeze()
