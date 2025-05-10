import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from utils import DecoderPrediction
from torch.nn import Conv1d
from .model_stack import ModelStack
from .common import get_padding


class ModelPair(torch.nn.Module):
    def __init__(self, model_config):
        super(ModelPair, self).__init__()
        self.model_config = model_config
        self.amp = ModelStack(model_config)
        self.phase = ModelStack(model_config)
        window = torch.hann_window(model_config.win_length)
        self.register_buffer("window", window, persistent=False)

        PSP_output_R_conv_kernel_size = 7
        PSP_output_I_conv_kernel_size = 7

        self.PSP_output_R_conv = Conv1d(
            model_config.n_fft // 2 + 1,
            model_config.n_fft // 2 + 1,
            PSP_output_R_conv_kernel_size,
            1,
            padding=get_padding(PSP_output_R_conv_kernel_size, 1),
        )
        self.PSP_output_I_conv = Conv1d(
            model_config.n_fft // 2 + 1,
            model_config.n_fft // 2 + 1,
            PSP_output_I_conv_kernel_size,
            1,
            padding=get_padding(PSP_output_I_conv_kernel_size, 1),
        )
        for m in [self.PSP_output_R_conv, self.PSP_output_I_conv]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, *, text, text_length, duration, mel, pitch, energy, text_mask):
        logamp = self.amp(
            text=text,
            text_length=text_length,
            duration=duration,
            mel=mel,
            pitch=pitch,
            energy=energy,
            text_mask=text_mask,
        )
        logamp = logamp.transpose(1, 2)
        logamp = F.pad(logamp, pad=(0, 1), mode="replicate")

        phase = self.phase(
            text=text,
            text_length=text_length,
            duration=duration,
            mel=mel,
            pitch=pitch,
            energy=energy,
            text_mask=text_mask,
        )
        phase = phase.transpose(1, 2)
        R = self.PSP_output_R_conv(phase)
        I = self.PSP_output_I_conv(phase)
        phase = torch.atan2(I, R)
        phase = F.pad(phase, pad=(0, 1), mode="replicate")

        rea = torch.exp(logamp) * torch.cos(phase)
        imag = torch.exp(logamp) * torch.sin(phase)
        spec = torch.complex(rea, imag)

        audio = torch.istft(
            spec,
            self.model_config.n_fft,
            hop_length=self.model_config.hop_length,
            win_length=self.model_config.win_length,
            window=self.window,
            center=True,
        )
        audio = torch.tanh(audio)
        return DecoderPrediction(
            audio=audio.unsqueeze(1),
            log_amplitude=logamp,
            phase=phase,
            real=rea,
            imaginary=imag,
        )
