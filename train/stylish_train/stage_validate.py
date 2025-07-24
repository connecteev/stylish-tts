import random
import torch
import torchaudio
from torch.nn import functional as F
from einops import rearrange

from loss_log import build_loss_log
from losses import compute_duration_ce_loss
from utils import length_to_mask, log_norm


@torch.no_grad()
def validate_alignment(batch, train):
    log = build_loss_log(train)
    mel = rearrange(batch.align_mel, "b f t -> b t f")
    ctc, _ = train.model.text_aligner(mel, batch.mel_length)
    train.stage.optimizer.zero_grad()

    loss_ctc = train.align_loss(
        ctc, batch.text, batch.mel_length, batch.text_length, step_type="eval"
    )

    blank = train.model_config.text_encoder.tokens
    logprobs = rearrange(ctc, "t b k -> b t k")
    confidence_total = 0.0
    confidence_count = 0
    for i in range(mel.shape[0]):
        _, scores = torchaudio.functional.forced_align(
            log_probs=logprobs[i].unsqueeze(0).contiguous(),
            targets=batch.text[i, : batch.text_length[i].item()].unsqueeze(0),
            input_lengths=batch.mel_length[i].unsqueeze(0),
            target_lengths=batch.text_length[i].unsqueeze(0),
            blank=blank,
        )
        confidence_total += scores.exp().sum()
        confidence_count += scores.shape[-1]
    log.add_loss("confidence", confidence_total / confidence_count)
    log.add_loss("align_loss", loss_ctc)
    return log, None, None, None


@torch.no_grad()
def validate_duration(batch, train):
    # state = BatchContext(train=train, model=train.model)
    # duration = state.predict_duration(batch)
    duration = train.model.duration_predictor(batch.text, batch.text_length)
    log = build_loss_log(train)
    loss_ce, loss_dur = compute_duration_ce_loss(
        duration,
        batch.alignment.sum(dim=-1),
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", loss_dur)

    return log.detach(), None, None, None


@torch.no_grad()
def validate_acoustic(batch, train):
    energy = log_norm(batch.mel.unsqueeze(1)).squeeze(1)
    pred_pitch, pred_energy = train.model.pitch_energy_predictor(
        batch.text, batch.text_length, batch.alignment
    )
    pred = train.model.speech_predictor(
        batch.text, batch.text_length, batch.alignment, batch.pitch, energy
    )
    # state = BatchContext(train=train, model=train.model)
    # pred = state.acoustic_prediction_single(batch)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
    )
    log.add_loss(
        "energy",
        torch.nn.functional.smooth_l1_loss(energy, pred_energy),
    )
    return log, batch.alignment[0], pred.audio, batch.audio_gt


@torch.no_grad()
def validate_textual(batch, train):
    pred_pitch, pred_energy = train.model.pitch_energy_predictor(
        batch.text, batch.text_length, batch.alignment
    )
    pred = train.model.speech_predictor(
        batch.text, batch.text_length, batch.alignment, pred_pitch, pred_energy
    )
    energy = log_norm(batch.mel.unsqueeze(1)).squeeze(1)
    # state = BatchContext(train=train, model=train.model)
    # pred = state.textual_prediction_single(batch)
    # energy = state.acoustic_energy(batch.mel)
    log = build_loss_log(train)
    train.stft_loss(pred.audio.squeeze(1), batch.audio_gt, log)
    log.add_loss(
        "pitch",
        torch.nn.functional.smooth_l1_loss(batch.pitch, pred_pitch),
    )
    log.add_loss("energy", torch.nn.functional.smooth_l1_loss(energy, pred_energy))
    # loss_ce, loss_dur = compute_duration_ce_loss(
    #     state.duration_prediction,
    #     batch.alignment.sum(dim=-1),
    #     batch.text_length,
    # )
    # log.add_loss("duration_ce", loss_ce)
    # log.add_loss("duration", loss_dur)
    return log, batch.alignment[0], pred.audio, batch.audio_gt
