import math
import random
from typing import Optional, Tuple
import torch
from torch.nn import functional as F
from einops import rearrange
from batch_context import BatchContext
from loss_log import LossLog, build_loss_log
from losses import compute_duration_ce_loss
from utils import length_to_mask, print_gpu_vram, log_norm


def train_alignment(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    log = build_loss_log(train)
    mel = rearrange(batch.align_mel, "b f t -> b t f")
    ctc, _ = model.text_aligner(mel, batch.mel_length)
    train.stage.optimizer.zero_grad()
    loss_ctc = train.align_loss(
        ctc, batch.text, batch.mel_length, batch.text_length, step_type="train"
    )

    log.add_loss(
        "align_loss",
        loss_ctc,
    )
    train.accelerator.backward(log.backwards_loss() * math.sqrt(batch.text.shape[0]))
    return log.detach(), None


def train_duration(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    # state = BatchContext(train=train, model=model)
    # duration = state.predict_duration(batch)
    duration = model.duration_predictor(batch.text, batch.text_length)
    train.stage.optimizer.zero_grad()
    log = build_loss_log(train)
    loss_ce, loss_dur = compute_duration_ce_loss(
        duration,
        batch.alignment.sum(dim=-1),
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", loss_dur)
    train.accelerator.backward(log.backwards_loss())

    return log.detach(), None


def train_acoustic(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    # state = BatchContext(train=train, model=model)
    with train.accelerator.autocast():
        print_gpu_vram("init")
        with torch.no_grad():
            energy = log_norm(batch.mel.unsqueeze(1)).squeeze(1)
        pred = model.speech_predictor(
            batch.text, batch.text_length, batch.alignment, batch.pitch, energy
        )
        pred_pitch, pred_energy = model.pitch_energy_predictor(
            batch.text, batch.text_length, batch.alignment
        )
        print_gpu_vram("predicted")
        train.stage.optimizer.zero_grad()

        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        print_gpu_vram("stft_loss")
        log.add_loss(
            "generator",
            train.generator_loss(
                batch.audio_gt.detach().unsqueeze(1).float(), pred.audio, ["mpd", "mrd"]
            ).mean(),
        )
        print_gpu_vram("generator_loss")
        log.add_loss(
            "slm",
            train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        )
        print_gpu_vram("slm_loss")
        if pred.magnitude is not None and pred.phase is not None:
            log.add_loss(
                "magphase",
                train.magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
            )
        print_gpu_vram("magphase_loss")

        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, pred_energy),
        )

        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )
        print_gpu_vram("backward")

    return log.detach(), pred.audio.detach()


def train_textual(
    batch, model, train, probing
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    # state = BatchContext(train=train, model=model)
    with train.accelerator.autocast():
        pred_pitch, pred_energy = model.pitch_energy_predictor(
            batch.text, batch.text_length, batch.alignment
        )
        pred = model.speech_predictor(
            batch.text, batch.text_length, batch.alignment, pred_pitch, pred_energy
        )
        with torch.no_grad():
            energy = log_norm(batch.mel.unsqueeze(1)).squeeze(1)
        # pred = state.textual_prediction_single(batch)
        # energy = state.acoustic_energy(batch.mel)
        # pitch = state.calculate_pitch(batch)
        train.stage.optimizer.zero_grad()
        log = build_loss_log(train)
        train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
        # log.add_loss(
        #     "generator",
        #     train.generator_loss(
        #         batch.audio_gt.detach().unsqueeze(1).float(), pred.audio, ["msbd"]
        #     ).mean(),
        # )
        # log.add_loss(
        #     "slm",
        #     train.wavlm_loss(batch.audio_gt.detach(), pred.audio),
        # )
        # if pred.magnitude is not None and pred.phase is not None:
        #     log.add_loss(
        #         "magphase",
        #         train.magphase_loss(pred.magnitude, pred.phase, batch.audio_gt),
        #     )
        log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
        )
        log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(energy, pred_energy),
        )
        # loss_ce, loss_dur = compute_duration_ce_loss(
        #     state.duration_prediction,
        #     batch.alignment.sum(dim=-1),
        #     batch.text_length,
        # )
        # log.add_loss("duration_ce", loss_ce)
        # log.add_loss("duration", loss_dur)
        train.accelerator.backward(
            log.backwards_loss() * math.sqrt(batch.text.shape[0])
        )

    return log.detach(), None
