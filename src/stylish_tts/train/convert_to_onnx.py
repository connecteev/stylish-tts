import torch
import torch.nn as nn
from stylish_tts.lib.config_loader import ModelConfig
from stylish_tts.lib.text_utils import TextCleaner
from stylish_tts.train.models.export_model import ExportModel
from stylish_tts.train.models.stft import STFT
from stylish_tts.train.utils import length_to_mask
import onnx
from torch.export.dynamic_shapes import Dim
from einops import rearrange


def add_meta_data_onnx(filename, key, value):
    model = onnx.load(filename)
    meta = model.metadata_props.add()
    meta.key = key
    meta.value = value
    onnx.save(model, filename)


def convert_to_onnx(
    model_config: ModelConfig,
    duration_path,
    speech_path,
    model_in,
    device,
    duration_processor,
):
    text_cleaner = TextCleaner(model_config.symbol)
    model = ExportModel(**model_in, device=device).eval()
    stft = STFT(
        filter_length=model_config.n_fft,
        hop_length=model_config.hop_length,
        win_length=model_config.win_length,
    )
    model.speech_predictor.generator.stft = stft.to(device).eval()
    duration_predictor = model_in.duration_predictor.eval()

    tokens = (
        torch.tensor(
            text_cleaner(
                "ðˈiːz wˈɜː tˈuː hˈæv ˈæn ɪnˈɔːɹməs ˈɪmpækt , nˈɑːt ˈoʊnliː bɪkˈɔz ðˈeɪ wˈɜː əsˈoʊsiːˌeɪtᵻd wˈɪð kˈɑːnstəntˌiːn ,"
            )
        )
        .unsqueeze(0)
        .to(device)
    )
    texts = torch.zeros([1, tokens.shape[1] + 2], dtype=int).to(device)
    texts[0, 1 : tokens.shape[1] + 1] = tokens
    texts = texts.long()
    text_lengths = torch.zeros([1], dtype=int).to(device)
    text_lengths[0] = tokens.shape[1] + 2
    text_mask = length_to_mask(text_lengths, text_lengths[0])

    with torch.no_grad():
        inputs = (texts, text_lengths)
        exported_program = torch.export.export(
            duration_predictor,
            inputs,
            dynamic_shapes=(
                (1, Dim.DYNAMIC),
                (1,),
            ),
        )
        onnx_program = torch.onnx.export(
            exported_program,
            inputs,
            opset_version=19,
            f=duration_path,
            input_names=["texts", "text_lengths"],
            output_names=["duration"],
            dynamo=True,
            optimize=False,
            dynamic_shapes={
                "texts": (1, Dim.DYNAMIC),
                "text_lengths": (1,),
            },
        )
        onnx_program.save(duration_path)

        dur_pred = duration_predictor(texts, text_lengths)
        dur_pred = rearrange(dur_pred, "1 k c -> k c")
        alignment = duration_processor(dur_pred, text_lengths).unsqueeze(0)

        inputs = (texts, text_lengths, alignment)

        exported_program = torch.export.export(
            model,
            inputs,
            dynamic_shapes=(
                (1, Dim.DYNAMIC),
                (1,),
                (1, Dim.DYNAMIC, Dim.DYNAMIC),
            ),
        )

        sample = exported_program.module().forward(texts, text_lengths, alignment)
        sample = sample.cpu().numpy()
        from scipy.io.wavfile import write
        import numpy as np

        sample = np.multiply(sample, 32768).astype(np.int16)
        write("sample_torch.wav", 24000, sample)

        onnx_program = torch.onnx.export(
            exported_program,
            inputs,
            opset_version=19,
            f=speech_path,
            input_names=["texts", "text_lengths", "alignment"],
            output_names=["waveform"],
            dynamo=True,
            optimize=False,
            dynamic_shapes=(
                (1, Dim.DYNAMIC),
                (1,),
                (1, Dim.DYNAMIC, Dim.DYNAMIC),
            ),
            # report=True,
        )
        onnx_program.save(speech_path)
    add_meta_data_onnx(speech_path, "model_config", model_config.model_dump_json())
