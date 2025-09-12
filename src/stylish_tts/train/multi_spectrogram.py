import torch
import torchaudio
from einops import rearrange


class Resolution:
    def __init__(self, *, fft, hop, window):
        self.fft = fft
        self.hop = hop
        self.window = window


resolutions = [
    # Resolution(fft=2048, hop=50, window=67),
    # Resolution(fft=2048, hop=100, window=127),
    Resolution(fft=2048, hop=50, window=240),  # 257),
    Resolution(fft=2048, hop=120, window=600),  # 509),
    Resolution(fft=2048, hop=240, window=1200),  # 1021),
    # Resolution(fft=2048, hop=100, window=2048),
]

multi_spectrogram_count = len(resolutions)


class MultiSpectrogram(torch.nn.Module):
    def __init__(
        self, resolutions=resolutions, window=torch.hann_window, *, sample_rate
    ):
        super(MultiSpectrogram, self).__init__()
        self.windows = [torch.hann_window(item.window) for item in resolutions]
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=128,
            sample_rate=sample_rate,
            n_stft=2048 // 2 + 1,
        )

    def calculate_single(self, audio, index, item):
        window = self.windows[index].to(audio.device)
        stft = torch.stft(
            audio,
            n_fft=item.fft,
            hop_length=item.hop,
            win_length=item.window,
            window=window,
            return_complex=True,
        )
        mag = torch.abs(stft)
        phase = (mag > 1e-3).detach() * torch.angle(stft)
        mag = torch.log1p(self.mel_scale(mag))
        mag = rearrange(mag, "b f t -> b 1 f t")
        return mag, phase

    def forward(self, *, target, pred):
        target_result = []
        pred_result = []
        target_phase_result = []
        pred_phase_result = []
        for index, item in enumerate(resolutions):
            with torch.no_grad():
                t_mag, t_phase = self.calculate_single(target, index, item)
                target_phase_result.append(t_phase)
                target_result.append(t_mag)
            p_mag, p_phase = self.calculate_single(pred, index, item)
            pred_phase_result.append(p_phase)
            pred_result.append(p_mag)
        return target_result, pred_result, target_phase_result, pred_phase_result
