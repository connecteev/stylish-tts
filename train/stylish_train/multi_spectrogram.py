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
        self.specs = torch.nn.ModuleList(
            [
                torchaudio.transforms.Spectrogram(
                    n_fft=item.fft,
                    win_length=item.window,
                    hop_length=item.hop,
                    window_fn=window,
                )
                for item in resolutions
            ]
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=128,
            sample_rate=sample_rate,
            n_stft=2048 // 2 + 1,
        )

    def forward(self, *, target, pred):
        target_result = []
        pred_result = []
        for spec in self.specs:
            with torch.no_grad():
                t = spec(target)
                t = torch.log(1 + self.mel_scale(t))
                # t = torch.pow(self.mel_scale(t), 0.3333)
                t = rearrange(t, "b f t -> b 1 f t")
            p = spec(pred)
            # p = torch.pow(self.mel_scale(p), 0.3333)
            p = torch.log(1 + self.mel_scale(p))
            p = rearrange(p, "b f t -> b 1 f t")
            target_result.append(t)
            pred_result.append(p)
        return target_result, pred_result
