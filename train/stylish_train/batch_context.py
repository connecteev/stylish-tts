import random
from typing import Optional

import torch
from torch.nn import functional as F
import torchaudio
from einops import rearrange, reduce
import train_context
from stylish_lib.config_loader import Config
from utils import length_to_mask, log_norm, print_gpu_vram


class BatchContext:
    def __init__(
        self,
        *,
        train: train_context.TrainContext,
        model,
    ):
        self.train: train_context.TrainContext = train
        self.config: Config = train.config
        # This is a subset containing only those models used this batch
        self.model = model

        self.pitch_prediction = None
        self.energy_prediction = None
        self.duration_prediction = None

    def text_encoding(self, texts: torch.Tensor, text_lengths: torch.Tensor):
        return self.model.text_encoder(texts, text_lengths)

    def text_duration_encoding(self, texts: torch.Tensor, text_lengths: torch.Tensor):
        return self.model.text_duration_encoder(texts, text_lengths)

    def acoustic_energy(self, mels: torch.Tensor):
        with torch.no_grad():
            energy = log_norm(mels.unsqueeze(1)).squeeze(1)
        return energy

    def calculate_pitch(self, batch, prediction=None):
        if prediction is None:
            prediction = batch.pitch
        return prediction

    def textual_style_embedding(self, sentence_embedding: torch.Tensor):
        return self.model.textual_style_encoder(sentence_embedding)

    def textual_prosody_embedding(self, sentence_embedding: torch.Tensor):
        return self.model.textual_prosody_encoder(sentence_embedding)

    def decoding(
        self,
        text_encoding,
        duration,
        pitch,
        energy,
        style,
        probing=False,
    ):
        mel, f0_curve = self.model.decoder(
            text_encoding @ duration, pitch, energy, style @ duration, probing=probing
        )
        print_gpu_vram("decoder")
        result = self.model.generator(
            mel=mel, style=style @ duration, pitch=f0_curve, energy=energy
        )
        print_gpu_vram("generator")
        return result

    def acoustic_prediction_single(self, batch, use_random_mono=True):
        text_encoding, _, _ = self.text_encoding(batch.text, batch.text_length)
        print_gpu_vram("text encoder")
        energy = self.acoustic_energy(batch.mel)
        style_embedding = self.textual_style_embedding(text_encoding)
        print_gpu_vram("style")
        pitch = self.calculate_pitch(batch).detach()
        prediction = self.decoding(
            text_encoding,
            batch.alignment,
            pitch,
            energy,
            style_embedding,
        )
        return prediction

    def textual_prediction_single(self, batch):
        text_encoding, _, _ = self.text_encoding(batch.text, batch.text_length)
        style_embedding = self.textual_style_embedding(text_encoding)
        self.duration_prediction = self.model.duration_predictor(
            batch.text, batch.text_length
        )
        self.pitch_prediction = self.model.pitch_predictor(
            batch.text, batch.mel_length, batch.alignment
        )
        self.energy_prediction = self.model.energy_predictor(
            batch.text, batch.mel_length, batch.alignment
        )
        pitch = self.calculate_pitch(batch, self.pitch_prediction)
        prediction = self.decoding(
            text_encoding,
            batch.alignment,
            pitch,
            self.energy_prediction,
            style_embedding,
        )
        return prediction
