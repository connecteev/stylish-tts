import pathlib, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import numpy
import torch
import librosa

from safetensors.torch import save_file
from dataprep.align_text import audio_list
import pyworld
import tqdm
from dataloader import get_frame_count, get_time_bin
from safetensors.torch import save_file


def calculate_pitch(config, model_config, method, workers):
    root = pathlib.Path(config.dataset.path)
    out = root / config.dataset.pitch_path
    wavdir = root / config.dataset.wav_path
    vals = calculate_pitch_set(
        "Val Set", method, root / config.dataset.val_data, wavdir, model_config, workers
    )
    trains = calculate_pitch_set(
        "Train set",
        method,
        root / config.dataset.train_data,
        wavdir,
        model_config,
        workers,
    )
    result = vals | trains
    save_file(result, out)


def calculate_pitch_set(label, method, path, wavdir, model_config, workers):
    if method == "rmvpe":
        # Initialize model?
        calculate_single = calculate_pitch_rmvpe
    elif method == "pyworld":
        calculate_single = calculate_pitch_pyworld
    else:
        exit("Invalid pitch calculation method passed")

    with path.open("r") as f:
        total_segments = sum(1 for _ in f)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        iterator = tqdm.tqdm(
            iterable=audio_list(path, wavdir, model_config),
            desc="Pitch prep " + label,
            unit="segments",
            initial=0,
            colour="GREEN",
            dynamic_ncols=True,
            total=total_segments,
        )
        for name, text_raw, wave in iterator:
            future_map[
                executor.submit(
                    calculate_single,
                    name,
                    text_raw,
                    wave,
                    model_config.sample_rate,
                    model_config.hop_length,
                )
            ] = name

        result = {}
        iterator = tqdm.tqdm(
            iterable=as_completed(future_map),
            desc="Pitch " + label,
            unit="segments",
            initial=0,
            colour="MAGENTA",
            dynamic_ncols=True,
            total=total_segments,
        )
        for future in iterator:
            name = future_map[future]
            try:
                current = future.result()
                if len(current.skipped) > 0:
                    print(f"{name} was skipped: {current.skipped}")
                else:
                    result[name] = current.tensor
            except Exception as e:
                print(f"{name} generated an exception: {str(e)}")
    return result


class PitchResult:
    def __init__(self):
        self.tensor = None
        self.skipped = ""


# def calculate_pitch_pyworld(path, wavdir, process_id):
def calculate_pitch_pyworld(name, text_raw, wave, sample_rate, hop_length):
    # result = {}
    # lines = path.read_text(encoding="utf-8").splitlines()

    # for count, line in enumerate(lines, 1):
    # fields = line.split("|")
    # name = fields[0]

    result = PitchResult()
    # if wave.shape[-1] == 2:
    #     wave = wave[:, 0].squeeze()
    time_bin = get_time_bin(wave.shape[0], hop_length)
    if time_bin == -1:
        result.skipped = "Too short"
        return result
    frame_count = get_frame_count(time_bin)
    pad_start = (frame_count * hop_length - wave.shape[0]) // 2
    pad_end = frame_count * hop_length - wave.shape[0] - pad_start
    wave = numpy.concatenate(
        [numpy.zeros([pad_start]), wave, numpy.zeros([pad_end])], axis=0
    )

    bad_f0 = 5
    zero_value = -10
    frame_period = hop_length / sample_rate * 1000
    f0, t = pyworld.harvest(wave, sample_rate, frame_period=frame_period)
    # if harvest fails, try dio
    if sum(f0 != 0) < bad_f0:
        f0, t = pyworld.dio(wave, sample_rate, frame_period=frame_period)
    pitch = pyworld.stonemask(wave, f0, t, sample_rate)
    pitch = torch.from_numpy(pitch).float().unsqueeze(0)
    if torch.any(torch.isnan(pitch)):
        pitch[torch.isnan(pitch)] = zero_value

    result.tensor = pitch
    return result


def calculate_pitch_rmvpe(path, wavdir, checkpoint, process_id):
    from rmvpe import RMVPE

    rmvpe = RMVPE(checkpoint)
    zero_value = -10
    result = {}
    lines = path.read_text(encoding="utf-8").splitlines()

    for count, line in enumerate(lines, 1):
        fields = line.split("|")
        name = fields[0]
        wave, sr = soundfile.read(wavdir / name)
        try:
            wave, sr = soundfile.read(wavdir / name)
            if sr != 24000:
                print(f"Skipping {name}: Wrong sample rate ({sr})")
        except:
            print(f"Skipping {name}: File not found or corrupted")
            continue
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        time_bin = get_time_bin(wave.shape[0])
        if time_bin == -1:
            print(f"Skipping {name}: Too short")
            continue
        frame_count = get_frame_count(time_bin)
        pad_start = (frame_count * 300 - wave.shape[0]) // 2
        pad_end = frame_count * 300 - wave.shape[0] - pad_start
        wave = numpy.concatenate(
            [numpy.zeros([pad_start]), wave, numpy.zeros([pad_end])], axis=0
        )

        wave_16k = librosa.resample(
            wave, orig_sr=24000, target_sr=16000, res_type="kaiser_best"
        )
        pitch_rmvpe = (
            torch.from_numpy(rmvpe.infer_from_audio(wave_16k)).float().unsqueeze(0)
        )  # (1, frames)
        pitch = torch.nn.functional.interpolate(
            pitch_rmvpe.unsqueeze(1),  # (1, 1, frames)
            size=frame_count,
            mode="linear",
            align_corners=True,
        ).squeeze(
            1
        )  # (1, frames)
        if torch.any(torch.isnan(pitch)):
            pitch[torch.isnan(pitch)] = zero_value
        # pitch = pitch[:, :-1]
        result[name] = pitch

        print(".", end=" ")
        if count % 100 == 0:
            print(f"P{process_id} {count}/{len(lines)}")
    return result
