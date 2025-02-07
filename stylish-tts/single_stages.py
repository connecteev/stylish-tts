import random, time, traceback
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from typing import List, Tuple, Any

from utils import length_to_mask, maximum_path, log_norm, log_print, get_image
from monotonic_align import mask_from_lens
from losses import magphase_loss
from config_loader import TrainContext


###############################################
# Helper Functions
###############################################

def prepare_batch(
    batch: List[Any], device: torch.device, keys_to_transfer: List[str] = None
) -> Tuple:
    """
    Transfers selected batch elements to the specified device.
    """
    if keys_to_transfer is None:
        keys_to_transfer = [
            "waves",
            "texts",
            "input_lengths",
            "ref_texts",
            "ref_lengths",
            "mels",
            "mel_input_length",
            "ref_mels",
        ]
    index = {
        "waves": 0,
        "texts": 1,
        "input_lengths": 2,
        "ref_texts": 3,
        "ref_lengths": 4,
        "mels": 5,
        "mel_input_length": 6,
        "ref_mels": 7,
    }
    prepared = tuple()
    for key in keys_to_transfer:
        if key not in index:
            raise ValueError(
                f"Key {key} not found in batch; valid keys: {list(index.keys())}"
            )
        prepared += (batch[index[key]].to(device),)
    return prepared


def compute_alignment(
    train: TrainContext,
    mels: torch.Tensor,
    texts: torch.Tensor,
    input_lengths: torch.Tensor,
    mel_input_length: torch.Tensor,
    apply_attention_mask: bool = False,
    use_random_choice: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the alignment used for training.
    Returns:
      - s2s_attn
      - s2s_attn_mono
      - s2s_pred
      - asr (encoded representation)
      - text_mask
      - mask (mel mask used for the aligner)
    """
    # Create masks.
    mask = length_to_mask(mel_input_length // (2**train.n_down)).to(
        train.config.training.device
    )
    text_mask = length_to_mask(input_lengths).to(train.config.training.device)

    # --- Text Aligner Forward Pass ---
    with train.accelerator.autocast():
        ppgs, s2s_pred, s2s_attn = train.model.text_aligner(mels, mask, texts)
        s2s_attn = s2s_attn.transpose(-1, -2)
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = s2s_attn.transpose(-1, -2)

    # Optionally apply extra attention mask.
    if apply_attention_mask:
        with torch.no_grad():
            attn_mask = (
                (~mask)
                .unsqueeze(-1)
                .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                .float()
                .transpose(-1, -2)
            )
            attn_mask = (
                attn_mask
                * (~text_mask)
                .unsqueeze(-1)
                .expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1])
                .float()
            )
            attn_mask = attn_mask < 1
        s2s_attn.masked_fill_(attn_mask, 0.0)

    # --- Monotonic Attention Path ---
    with torch.no_grad():
        mask_ST = mask_from_lens(
            s2s_attn, input_lengths, mel_input_length // (2**train.n_down)
        )
        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

    # --- Text Encoder Forward Pass ---
    with train.accelerator.autocast():
        t_en = train.model.text_encoder(texts, input_lengths, text_mask)
        if use_random_choice:
            asr = t_en @ (s2s_attn if bool(random.getrandbits(1)) else s2s_attn_mono)
        else:
            asr = t_en @ s2s_attn_mono

    return s2s_attn, s2s_attn_mono, s2s_pred, asr, text_mask, mask


def compute_duration_ce_loss(
    s2s_preds: List[torch.Tensor],
    text_inputs: List[torch.Tensor],
    text_lengths: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the duration and binary cross-entropy losses over a batch.
    Returns (loss_ce, loss_dur).
    """
    loss_ce = 0
    loss_dur = 0
    for pred, inp, length in zip(s2s_preds, text_inputs, text_lengths):
        pred = pred[:length, :]
        inp = inp[:length].long()
        target = torch.zeros_like(pred)
        for i in range(target.shape[0]):
            target[i, : inp[i]] = 1
        dur_pred = torch.sigmoid(pred).sum(dim=1)
        loss_dur += F.l1_loss(dur_pred[1 : length - 1], inp[1 : length - 1])
        loss_ce += F.binary_cross_entropy_with_logits(pred.flatten(), target.flatten())
    n = len(text_lengths)
    return loss_ce / n, loss_dur / n


def scale_gradients(model: dict, thresh: float, scale: float) -> None:
    """
    Scales (and clips) gradients for the given model dictionary.
    """
    total_norm = {}
    for key in model.keys():
        total_norm[key] = 0.0
        parameters = [
            p for p in model[key].parameters() if p.grad is not None and p.requires_grad
        ]
        for p in parameters:
            total_norm[key] += p.grad.detach().data.norm(2).item() ** 2
        total_norm[key] = total_norm[key] ** 0.5
    if total_norm.get("predictor", 0) > thresh:
        for key in model.keys():
            for p in model[key].parameters():
                if p.grad is not None:
                    p.grad *= 1 / total_norm["predictor"]
    # Apply additional scaling to specific modules.
    for p in model["predictor"].duration_proj.parameters():
        if p.grad is not None:
            p.grad *= scale
    for p in model["predictor"].lstm.parameters():
        if p.grad is not None:
            p.grad *= scale
    for p in model["diffusion"].parameters():
        if p.grad is not None:
            p.grad *= scale


def optimizer_step(train: TrainContext, keys: List[str]) -> None:
    """
    Steps the optimizer for each module key in keys.
    """
    for key in keys:
        train.optimizer.step(key)


def log_and_save_checkpoint(
    train: TrainContext, current_step: int, prefix: str = "epoch_1st"
) -> None:
    """
    Logs metrics and saves a checkpoint.
    """
    state = {
        "net": {key: train.model[key].state_dict() for key in train.model},
        "optimizer": train.optimizer.state_dict(),
        "iters": train.manifest.iters,
        "val_loss": train.best_loss,
        "epoch": train.manifest.current_epoch,
    }
    if current_step == -1:
        filename = f"{prefix}_{train.manifest.current_epoch:05d}.pth"
    else:
        filename = (
            f"{prefix}_{train.manifest.current_epoch:05d}_step_{current_step:09d}.pth"
        )
    save_path = osp.join(train.config.training.out_dir, filename)
    torch.save(state, save_path)
    print(f"Saving checkpoint to {save_path}")


###############################################
# Individual Loss Functions (First Stage)
###############################################

def loss_mel_first_stage(train, batch):
    """
    Computes only the mel/stft-based loss for the first stage.
    """
    # Minimal data needed
    texts, input_lengths, mels, mel_input_length = prepare_batch(
        batch,
        train.config.training.device,
        ["texts", "input_lengths", "mels", "mel_input_length"],
    )
    if mels.shape[-1] < 40:
        # Return None to indicate "skip this batch"
        return None

    with torch.no_grad():
        real_norm = log_norm(mels.unsqueeze(1)).squeeze(1)
        F0_real, _, _ = train.model.pitch_extractor(mels.unsqueeze(1))

    # Possibly alignment to get ASR features
    s2s_attn, s2s_attn_mono, s2s_pred, asr, _, _ = compute_alignment(
        train,
        mels,
        texts,
        input_lengths,
        mel_input_length,
        apply_attention_mask=True,
        use_random_choice=False,
    )

    style_emb = train.model.style_encoder(mels.unsqueeze(1))
    # ---- Autocast for the forward pass + STFT loss ----
    with train.accelerator.autocast():
        y_rec, mag_rec, phase_rec = train.model.decoder(asr, F0_real, real_norm, style_emb)
        loss_mel = train.stft_loss(y_rec.squeeze(), 
                                   prepare_batch(batch, train.config.training.device, ["waves"])[0])

    return loss_mel


def loss_mono_first_stage(train, batch):
    """
    Example: monotonic alignment loss only. 
    (For the 'first_tma' scenario.)
    """
    texts, input_lengths, mels, mel_input_length = prepare_batch(
        batch,
        train.config.training.device,
        ["texts", "input_lengths", "mels", "mel_input_length"],
    )
    if mels.shape[-1] < 40:
        return None

    s2s_attn, s2s_attn_mono, s2s_pred, asr, _, _ = compute_alignment(
        train,
        mels,
        texts,
        input_lengths,
        mel_input_length,
        apply_attention_mask=True,
        use_random_choice=False,
    )

    # monotonic alignment loss can be L1(s2s_attn, s2s_attn_mono)
    with train.accelerator.autocast():
        loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10.0
    return loss_mono


def loss_s2s_first_stage(train, batch):
    """
    Example: s2s cross-entropy only (for text alignment).
    """
    texts, input_lengths, mels, mel_input_length = prepare_batch(
        batch,
        train.config.training.device,
        ["texts", "input_lengths", "mels", "mel_input_length"],
    )
    if mels.shape[-1] < 40:
        return None

    s2s_attn, s2s_attn_mono, s2s_pred, asr, _, _ = compute_alignment(
        train,
        mels,
        texts,
        input_lengths,
        mel_input_length,
        apply_attention_mask=True,
        use_random_choice=False,
    )

    with train.accelerator.autocast():
        loss_s2s = 0
        for pred, txt, length in zip(s2s_pred, texts, input_lengths):
            loss_s2s += F.cross_entropy(pred[:length], txt[:length])
        loss_s2s /= texts.size(0)
    return loss_s2s


def loss_gen_first_stage(train, batch):
    """
    Example: generator adversarial loss (1st stage).
    """
    texts, input_lengths, mels, mel_input_length = prepare_batch(
        batch,
        train.config.training.device,
        ["texts", "input_lengths", "mels", "mel_input_length"],
    )
    if mels.shape[-1] < 40:
        return None

    with torch.no_grad():
        real_norm = log_norm(mels.unsqueeze(1)).squeeze(1)
        F0_real, _, _ = train.model.pitch_extractor(mels.unsqueeze(1))

    s2s_attn, s2s_attn_mono, s2s_pred, asr, _, _ = compute_alignment(
        train, mels, texts, input_lengths, mel_input_length
    )
    style_emb = train.model.style_encoder(mels.unsqueeze(1))
    with train.accelerator.autocast():
        y_rec, _, _ = train.model.decoder(asr, F0_real, real_norm, style_emb)
        wav = prepare_batch(batch, train.config.training.device, ["waves"])[0]
        wav.requires_grad_(False)
        # Example generator loss (from your code: train.gl is a module that expects (wav, y_rec))
        loss_g = train.gl(wav.unsqueeze(1).detach(), y_rec).mean()
    return loss_g


def loss_slm_first_stage(train, batch):
    """
    Example: slm (WavLM) feature matching loss only.
    """
    texts, input_lengths, mels, mel_input_length = prepare_batch(
        batch,
        train.config.training.device,
        ["texts", "input_lengths", "mels", "mel_input_length"],
    )
    if mels.shape[-1] < 40:
        return None

    with torch.no_grad():
        real_norm = log_norm(mels.unsqueeze(1)).squeeze(1)
        F0_real, _, _ = train.model.pitch_extractor(mels.unsqueeze(1))

    s2s_attn, s2s_attn_mono, s2s_pred, asr, _, _ = compute_alignment(
        train, mels, texts, input_lengths, mel_input_length
    )
    style_emb = train.model.style_encoder(mels.unsqueeze(1))
    with train.accelerator.autocast():
        y_rec, _, _ = train.model.decoder(asr, F0_real, real_norm, style_emb)
        wav = prepare_batch(batch, train.config.training.device, ["waves"])[0]
        wav.requires_grad_(False)
        loss_slm = train.wl(wav.detach(), y_rec)
    return loss_slm


def get_stage_loss_dict(train: TrainContext):
    """
    Return a {loss_name: weight} dictionary, 
    depending on whether stage is first, first_tma, second, second_style, second_joint, etc.
    """
    # Shortcuts for easier referencing
    lw = train.config.loss_weight
    stage = train.manifest.stage

    # We'll return only the subset relevant to each stage
    if stage == "first":
        # e.g. only mel, gen, slm
        return {
            "mel": lw.mel,
            "gen": lw.gen,
            "slm": lw.slm
        }
    elif stage == "first_tma":
        # e.g. includes monotonic alignment and s2s
        return {
            "mel": lw.mel,
            "gen": lw.gen,
            "slm": lw.slm,
            "mono": lw.mono,
            "s2s": lw.s2s
        }
    elif stage == "second":
        # e.g. includes F0, norm, duration, style, diffusion, etc.
        return {
            "mel": lw.mel,
            "F0": lw.F0,
            "norm": lw.norm,
            "duration": lw.duration,
            "duration_ce": lw.duration_ce,
            "style": lw.style,
            "diffusion": lw.diffusion,
            # possibly also gen, slm if you want
        }
    # Likewise for "second_style", "second_joint", etc.
    elif stage == "second_style":
        return {
            "mel": lw.mel,
            "F0": lw.F0,
            "norm": lw.norm,
            "duration": lw.duration,
            "duration_ce": lw.duration_ce,
            "style": lw.style,
            "diffusion": lw.diffusion
        }
    elif stage == "second_joint":
        return {
            "mel": lw.mel,
            "F0": lw.F0,
            "norm": lw.norm,
            "duration": lw.duration,
            "duration_ce": lw.duration_ce,
            "style": lw.style,
            "diffusion": lw.diffusion,
            # plus any extras you want for "joint"
        }
    else:
        # fallback (maybe just return everything or raise an error)
        return {
            "mel": lw.mel,
            "gen": lw.gen,
            "slm": lw.slm,
            "mono": lw.mono,
            "s2s": lw.s2s,
            "F0": lw.F0,
            "norm": lw.norm,
            "duration": lw.duration,
            "duration_ce": lw.duration_ce,
            "style": lw.style,
            "diffusion": lw.diffusion,
        }

###############################################
# Single-Loss "train_first" Implementation
###############################################

def train_first_single_loss(
    i: int, 
    batch, 
    running_loss: float, 
    iters: int, 
    train: TrainContext
) -> Tuple[float, int]:
    """
    Single-loss approach for the first stage.
    We pick one sub-loss from a weighted list (external logic),
    compute that sub-loss, then step the corresponding modules.
    """

    # ------------------------------------------------
    # 1) Pick which loss to compute (sample from weighted list)
    # ------------------------------------------------
    if train.loss_index >= len(train.weighted_list):
        random.shuffle(train.weighted_list)
        train.loss_index = 0
    selected_loss = train.weighted_list[train.loss_index]
    train.loss_index += 1

    # Zero grads
    train.optimizer.zero_grad()

    # ------------------------------------------------
    # 2) Dispatch to the appropriate sub-loss function
    # ------------------------------------------------
    if selected_loss == "mel":
        loss_val = loss_mel_first_stage(train, batch)
        modules_to_step = ["text_encoder", "style_encoder", "decoder"]
    elif selected_loss == "mono":
        loss_val = loss_mono_first_stage(train, batch)
        # Typically you might step text_aligner if you're training alignment
        modules_to_step = ["text_aligner"]
    elif selected_loss == "s2s":
        loss_val = loss_s2s_first_stage(train, batch)
        modules_to_step = ["text_aligner"]
    elif selected_loss == "gen":
        loss_val = loss_gen_first_stage(train, batch)
        modules_to_step = ["text_encoder", "style_encoder", "decoder", "mpd", "msd"]
    elif selected_loss == "slm":
        loss_val = loss_slm_first_stage(train, batch)
        modules_to_step = ["decoder", "style_encoder"]
    else:
        # Default: no-op
        loss_val = None
        modules_to_step = []

    # If batch was too short or some error => skip
    if loss_val is None:
        return running_loss, iters

    # ------------------------------------------------
    # 3) Backprop + optimizer step
    # ------------------------------------------------
    train.accelerator.backward(loss_val)
    optimizer_step(train, modules_to_step)

    # Logging: update counters
    running_loss += loss_val.item()
    train.manifest.iters += 1
    if train.accelerator.is_main_process and ((i+1) % train.config.training.log_interval == 0):
        avg_loss = running_loss / train.config.training.log_interval
        train.logger.info(
            f"[Iter {i+1}] single-loss={selected_loss}, avg_loss={avg_loss:.5f}"
        )
        train.writer.add_scalar(f"train/{selected_loss}_loss", avg_loss, train.manifest.iters)
        running_loss = 0
        print("Time elapsed:", time.time() - train.start_time)

    # If you want to do your usual val_interval / save_interval checks, you can do so:
    if (i + 1) % train.config.training.val_interval == 0 or (
        i + 1
    ) % train.config.training.save_interval == 0:
        save = (i + 1) % train.config.training.save_interval == 0
        train.validate(current_step=i + 1, save=save, train=train)

    return running_loss, iters


###############################################
# Example of Single-Loss Functions (Second Stage)
###############################################

def loss_mel_second_stage(train, batch):
    """
    Minimal example: decode -> STFT loss
    """
    # Basic prep
    waves, texts, input_lengths, mels, mel_input_length = prepare_batch(
        batch,
        train.config.training.device,
        ["waves","texts","input_lengths","mels","mel_input_length"]
    )
    if mels.shape[-1] < 40:
        return None

    wav = waves.unsqueeze(1)

    # alignment
    _, s2s_attn_mono, _, asr, text_mask, _ = compute_alignment(
        train,
        mels,
        texts,
        input_lengths,
        mel_input_length,
        apply_attention_mask=False,
        use_random_choice=False,
    )
    with torch.no_grad():
        F0_real, _, _ = train.model.pitch_extractor(mels.unsqueeze(1))
        N_real = log_norm(mels.unsqueeze(1)).squeeze(1)
        gs = train.model.style_encoder(mels.unsqueeze(1))

    with train.accelerator.autocast():
        y_rec, _, _ = train.model.decoder(asr, F0_real, N_real, gs)
        loss_mel = train.stft_loss(y_rec, wav.detach())

    return loss_mel


def loss_diff_second_stage(train, batch):
    """
    Example: style diffusion sub-loss
    """
    # Basic prep
    waves, texts, input_lengths, mels, mel_input_length = prepare_batch(
        batch,
        train.config.training.device,
        ["waves","texts","input_lengths","mels","mel_input_length"]
    )
    if mels.shape[-1] < 40:
        return None

    # We'll do s_dur/gs inside autocast
    with train.accelerator.autocast():
        s_dur = train.model.predictor_encoder(mels.unsqueeze(1))
        gs = train.model.style_encoder(mels.unsqueeze(1))
        s_trg = torch.cat([gs, s_dur], dim=-1).detach()

        bert_dur = train.model.bert(texts, attention_mask=None)
        d_en = train.model.bert_encoder(bert_dur).transpose(-1, -2)

        num_steps = np.random.randint(3, 5)
        noise = torch.randn_like(s_trg).unsqueeze(1).to(train.config.training.device)

        if train.config.model.multispeaker:
            loss_diff = train.model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=None).mean()
        else:
            loss_diff = train.model.diffusion.module.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean()

    return loss_diff


###############################################
# Single-Loss "train_second" Implementation
###############################################

def train_second_single_loss(
    i: int, 
    batch, 
    running_loss: float, 
    iters: int, 
    train: TrainContext
) -> Tuple[float, int]:
    """
    Single-loss approach for the second stage.
    We pick one sub-loss from a weighted list, compute that sub-loss,
    and step the relevant modules.
    """

    # ------------------------------------------------
    # 1) Pick which loss to compute (sample from weighted list)
    # ------------------------------------------------
    if train.loss_index >= len(train.weighted_list):
        random.shuffle(train.weighted_list)
        train.loss_index = 0
    selected_loss = train.weighted_list[train.loss_index]
    train.loss_index += 1

    train.optimizer.zero_grad()

    # Dispatch to sub-loss function
    if selected_loss == "mel":
        loss_val = loss_mel_second_stage(train, batch)
        # Possibly step "decoder", "style_encoder", "predictor"...
        modules_to_step = ["decoder", "style_encoder", "predictor"]
    elif selected_loss == "diff":
        loss_val = loss_diff_second_stage(train, batch)
        modules_to_step = ["diffusion", "predictor", "predictor_encoder"]
    else:
        # You can define more: "loss_dur_second_stage", "loss_gen_second_stage", etc.
        loss_val = None
        modules_to_step = []

    if loss_val is None:
        return running_loss, iters

    train.accelerator.backward(loss_val)
    optimizer_step(train, modules_to_step)

    running_loss += loss_val.item()
    train.manifest.iters += 1

    if train.accelerator.is_main_process and ((i+1) % train.config.training.log_interval == 0):
        avg_loss = running_loss / train.config.training.log_interval
        train.logger.info(
            f"[Iter {i+1}] single-loss={selected_loss}, avg_loss={avg_loss:.5f}"
        )
        train.writer.add_scalar(f"train/{selected_loss}_loss", avg_loss, train.manifest.iters)
        running_loss = 0
        print("Time elapsed:", time.time() - train.start_time)

    # If you want val/save intervals:
    if (i + 1) % train.config.training.gival_interval == 0 or (
        i + 1
    ) % train.config.training.save_interval == 0:
        save = (i + 1) % train.config.training.save_interval == 0
        train.validate(current_step=i + 1, save=save, train=train)

    return running_loss, iters


###############################################
# validate_first
###############################################

def validate_first(current_step: int, save: bool, train: TrainContext) -> None:
    """
    Validation function for the first stage.
    (Mostly unchanged from your original code.)
    """
    loss_test = 0
    # Set models to evaluation mode.
    for key in train.model:
        train.model[key].eval()

    with torch.no_grad():
        iters_test = 0
        for batch in train.val_dataloader:
            waves, texts, input_lengths, mels, mel_input_length = prepare_batch(
                batch,
                train.config.training.device,
                ["waves", "texts", "input_lengths", "mels", "mel_input_length"],
            )
            mask = length_to_mask(mel_input_length // (2**train.n_down)).to(
                train.config.training.device
            )
            text_mask = length_to_mask(input_lengths).to(train.config.training.device)
            _, _, s2s_attn = train.model.text_aligner(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)
            mask_ST = mask_from_lens(
                s2s_attn, input_lengths, mel_input_length // (2**train.n_down)
            )
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
            t_en = train.model.text_encoder(texts, input_lengths, text_mask)
            asr = t_en @ s2s_attn_mono

            if mels.shape[-1] < 40 or (
                mels.shape[-1] < 80
                and not train.config.embedding_encoder.skip_downsamples
            ):
                log_print("Skipping batch. TOO SHORT", train.logger)
                continue

            F0_real, _, _ = train.model.pitch_extractor(mels.unsqueeze(1))
            s = train.model.style_encoder(mels.unsqueeze(1))
            real_norm = log_norm(mels.unsqueeze(1)).squeeze(1)
            y_rec, _, _ = train.model.decoder(asr, F0_real, real_norm, s)
            loss_mel = train.stft_loss(y_rec.squeeze(), waves.detach())
            loss_test += loss_mel.item()
            iters_test += 1

    if train.accelerator.is_main_process:
        avg_loss = loss_test / iters_test if iters_test > 0 else float("inf")
        print(
            f"Epochs:{train.manifest.current_epoch} Steps:{current_step} Loss:{avg_loss} Best_Loss:{train.best_loss}"
        )
        log_print(
            f"Epochs:{train.manifest.current_epoch} Steps:{current_step} Loss:{avg_loss} Best_Loss:{train.best_loss}",
            train.logger,
        )
        log_print(f"Validation loss: {avg_loss:.3f}\n\n\n\n", train.logger)
        train.writer.add_scalar("eval/mel_loss", avg_loss, train.manifest.current_epoch)
        attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
        train.writer.add_figure("eval/attn", attn_image, train.manifest.current_epoch)

        with torch.no_grad():
            for bib in range(min(len(asr), 6)):
                mel_length = int(mel_input_length[bib].item())
                gt = mels[bib, :, :mel_length].unsqueeze(0)
                en = asr[bib, :, : mel_length // 2].unsqueeze(0)
                F0_real, _, _ = train.model.pitch_extractor(gt.unsqueeze(1))
                s = train.model.style_encoder(gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                y_rec, _, _ = train.model.decoder(en, F0_real, real_norm, s)
                train.writer.add_audio(
                    f"eval/y{bib}",
                    y_rec.cpu().numpy().squeeze(),
                    train.manifest.current_epoch,
                    sample_rate=train.config.preprocess.sample_rate,
                )
                if train.manifest.current_epoch == 0:
                    train.writer.add_audio(
                        f"gt/y{bib}",
                        waves[bib].squeeze(),
                        train.manifest.current_epoch,
                        sample_rate=train.config.preprocess.sample_rate,
                    )

        if (
            train.manifest.current_epoch % train.config.training.save_epoch_interval
            == 0
            and save
            and current_step == -1
        ):
            if avg_loss < train.best_loss:
                train.best_loss = avg_loss
            print("Saving..")
            log_and_save_checkpoint(train, current_step, prefix="epoch_1st")
        if save and current_step != -1:
            if avg_loss < train.best_loss:
                train.best_loss = avg_loss
            print("Saving..")
            log_and_save_checkpoint(train, current_step, prefix="epoch_1st")

    for key in train.model:
        train.model[key].train()


###############################################
# validate_second
###############################################

def validate_second(current_step: int, save: bool, train: TrainContext) -> None:
    """
    Validation function for the second stage.
    (Mostly unchanged from your original code.)
    """
    loss_test = 0
    loss_align = 0
    loss_f = 0
    for key in train.model:
        train.model[key].eval()

    with torch.no_grad():
        iters_test = 0
        for batch in train.val_dataloader:
            try:
                waves, texts, input_lengths, mels, mel_input_length, ref_mels = (
                    prepare_batch(
                        batch,
                        train.config.training.device,
                        [
                            "waves",
                            "texts",
                            "input_lengths",
                            "mels",
                            "mel_input_length",
                            "ref_mels",
                        ],
                    )
                )
                mask = length_to_mask(mel_input_length // (2**train.n_down)).to(
                    train.config.training.device
                )
                text_mask = length_to_mask(input_lengths).to(
                    train.config.training.device
                )
                _, _, s2s_attn = train.model.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)
                mask_ST = mask_from_lens(
                    s2s_attn, input_lengths, mel_input_length // (2**train.n_down)
                )
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                t_en = train.model.text_encoder(texts, input_lengths, text_mask)
                asr = t_en @ s2s_attn_mono
                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                if mels.shape[-1] < 40 or (
                    mels.shape[-1] < 80
                    and not train.config.embedding_encoder.skip_downsamples
                ):
                    log_print("Skipping batch. TOO SHORT", train.logger)
                    continue

                s = train.model.predictor_encoder(mels.unsqueeze(1))
                gs = train.model.style_encoder(mels.unsqueeze(1))
                s_trg = torch.cat([s, gs], dim=-1).detach()
                bert_dur = train.model.bert(texts, attention_mask=(~text_mask).int())
                d_en = train.model.bert_encoder(bert_dur).transpose(-1, -2)
                d, p = train.model.predictor(
                    (d_en, s, input_lengths, s2s_attn_mono, text_mask),
                    predict_F0N=False,
                )
                F0_fake, N_fake = train.model.predictor((p, s), predict_F0N=True)
                loss_dur = 0
                for pred, inp, length in zip(d, d_gt, input_lengths):
                    pred = pred[:length, :]
                    inp = inp[:length].long()
                    target = torch.zeros_like(pred)
                    for i in range(target.shape[0]):
                        target[i, : inp[i]] = 1
                    dur_pred = torch.sigmoid(pred).sum(dim=1)
                    loss_dur += F.l1_loss(dur_pred[1 : length - 1], inp[1 : length - 1])
                loss_dur /= texts.size(0)

                y_rec, _, _ = train.model.decoder(asr, F0_fake, N_fake, gs)
                loss_mel = train.stft_loss(y_rec.squeeze(1), waves.detach())

                F0_real, _, _ = train.model.pitch_extractor(mels.unsqueeze(1))
                loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                loss_test += loss_mel.mean()
                loss_align += loss_dur.mean()
                loss_f += loss_F0.mean()
                iters_test += 1

            except Exception as e:
                print(f"Encountered exception: {e}")
                traceback.print_exc()
                continue

    if train.accelerator.is_main_process:
        avg_loss = loss_test / iters_test if iters_test > 0 else float("inf")
        print(
            f"Epochs: {train.manifest.current_epoch}, Steps: {current_step}, Loss: {avg_loss}, Best_Loss: {train.best_loss}"
        )
        train.logger.info(
            f"Validation loss: {avg_loss:.3f}, Dur loss: {loss_align / iters_test:.3f}, F0 loss: {loss_f / iters_test:.3f}\n\n\n"
        )
        train.writer.add_scalar("eval/mel_loss", avg_loss, train.manifest.current_epoch)
        attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
        train.writer.add_figure("eval/attn", attn_image, train.manifest.current_epoch)

        with torch.no_grad():
            for bib in range(min(len(asr), 6)):
                mel_length = int(mel_input_length[bib].item())
                gt = mels[bib, :, :mel_length].unsqueeze(0)
                en = asr[bib, :, : mel_length // 2].unsqueeze(0)
                F0_real, _, _ = train.model.pitch_extractor(gt.unsqueeze(1))
                s = train.model.style_encoder(gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                y_rec, _, _ = train.model.decoder(en, F0_real, real_norm, s)
                train.writer.add_audio(
                    f"eval/y{bib}",
                    y_rec.cpu().numpy().squeeze(),
                    train.manifest.current_epoch,
                    sample_rate=train.config.preprocess.sample_rate,
                )
                if train.manifest.current_epoch == 0:
                    train.writer.add_audio(
                        f"gt/y{bib}",
                        waves[bib].squeeze(),
                        train.manifest.current_epoch,
                        sample_rate=train.config.preprocess.sample_rate,
                    )

        if (
            train.manifest.current_epoch % train.config.training.save_epoch_interval
            == 0
            and save
            and current_step == -1
        ):
            if avg_loss < train.best_loss:
                train.best_loss = avg_loss
            print("Saving..")
            log_and_save_checkpoint(train, current_step, prefix="epoch_2nd")
        if save and current_step != -1:
            if avg_loss < train.best_loss:
                train.best_loss = avg_loss
            print("Saving..")
            log_and_save_checkpoint(train, current_step, prefix="epoch_2nd")

    for key in train.model:
        train.model[key].train()
