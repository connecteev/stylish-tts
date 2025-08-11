# coding:utf-8


import math

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from stylish_lib.config_loader import ModelConfig


from .text_aligner import tdnn_blstm_ctc_model_base

from .discriminators.multi_period import MultiPeriodDiscriminator
from .discriminators.multi_resolution import MultiResolutionDiscriminator
from .discriminators.multi_subband import MultiScaleSubbandCQTDiscriminator

from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor

from .text_encoder import TextEncoder
from .fine_style_encoder import CoarseStyleEncoder
from .style_encoder import StyleEncoder
from .decoder import Decoder
from .ringformer import RingformerGenerator
from .pitch_energy_predictor import PitchEnergyPredictor
from .speech_predictor import SpeechPredictor

from munch import Munch
import safetensors
from huggingface_hub import hf_hub_download

import logging

logger = logging.getLogger(__name__)


def build_model(model_config: ModelConfig):
    text_aligner = tdnn_blstm_ctc_model_base(
        model_config.n_mels, model_config.text_encoder.tokens
    )

    duration_predictor = DurationPredictor(
        style_dim=model_config.style_dim,
        inter_dim=model_config.inter_dim,
        text_config=model_config.text_encoder,
        style_config=model_config.style_encoder,
        duration_config=model_config.duration_predictor,
    )

    pitch_energy_predictor = PitchEnergyPredictor(
        style_dim=model_config.style_dim,
        inter_dim=model_config.inter_dim,
        text_config=model_config.text_encoder,
        style_config=model_config.style_encoder,
        duration_config=model_config.duration_predictor,
        pitch_energy_config=model_config.pitch_energy_predictor,
    )

    pe_text_encoder = TextEncoder(inter_dim=512, config=model_config.text_encoder)
    pe_text_style_encoder = CoarseStyleEncoder(512, 64, model_config.style_encoder)
    pe_mel_style_encoder = StyleEncoder(80, 64, 384, True)

    nets = Munch(
        text_aligner=text_aligner,
        duration_predictor=duration_predictor,
        pitch_energy_predictor=pitch_energy_predictor,
        speech_predictor=SpeechPredictor(model_config),
        mpd=MultiPeriodDiscriminator(),
        msbd=MultiScaleSubbandCQTDiscriminator(sample_rate=model_config.sample_rate),
        mrd=MultiResolutionDiscriminator(),
        pe_text_encoder=pe_text_encoder,
        pe_text_style_encoder=pe_text_style_encoder,
        pe_mel_style_encoder=pe_mel_style_encoder,
    )

    return nets
