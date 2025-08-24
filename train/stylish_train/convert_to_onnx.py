import torch
import torch.nn as nn
from stylish_lib.config_loader import ModelConfig
from stylish_lib.text_utils import TextCleaner
from models.export_model import ExportModel
from models.stft import STFT
from utils import length_to_mask
import onnx
from torch.export.dynamic_shapes import Dim
from onnx_diagnostic.torch_export_patches import torch_export_patches


def add_meta_data_onnx(filename, key, value):
    model = onnx.load(filename)
    meta = model.metadata_props.add()
    meta.key = key
    meta.value = value
    onnx.save(model, filename)


def convert_to_onnx(
    model_config: ModelConfig, out_dir, model_in, device, duration_processor
):
    import ai_edge_torch

    text_cleaner = TextCleaner(model_config.symbol)
    model = ExportModel(**model_in, device=device).eval()
    stft = STFT(
        filter_length=model_in.speech_predictor.generator.gen_istft_n_fft,
        hop_length=model_in.speech_predictor.generator.gen_istft_hop_size,
        win_length=model_in.speech_predictor.generator.gen_istft_n_fft,
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
        with torch_export_patches():  # stop_if_static=1, verbose=1):
            filename = f"{out_dir}/duration.tflite"
            inputs = (texts, text_lengths)
            # exported_program = torch.export.export(
            #     duration_predictor,
            #     inputs,
            #     dynamic_shapes=(
            #         (1, Dim("tokens", max=1000)),
            #         (1,),
            #     ),
            # )
            # onnx_program = torch.onnx.export(
            #     exported_program,
            #     # model_in.duration_predictor,
            #     inputs,
            #     opset_version=21,
            #     f=filename,
            #     input_names=["texts", "text_lengths"],
            #     output_names=["duration"],
            #     dynamo=True,
            #     optimize=False,
            #     dynamic_shapes={
            #         "texts": (1, Dim("tokens", max=1000)),
            #         "text_lengths": (1,),
            #     },
            # )
            # onnx_program.save(filename)
            edge_model = ai_edge_torch.convert(duration_predictor, inputs)
            edge_model.export(filename)

            dur_pred = duration_predictor(texts, text_lengths)
            # dur_pred = rearrange(dur_pred, "1 k c -> k c")
            dur_pred = dur_pred.squeeze(0)
            alignment = duration_processor(dur_pred, text_lengths).unsqueeze(0)

            filename = f"{out_dir}/stylish.tflite"
            inputs = (texts, text_lengths, alignment)
            edge_model = ai_edge_torch.convert(model, inputs)
            edge_model.export(filename)
            # torch.onnx.export(
            #     model,
            #     inputs,
            #     opset_version=14,
            #     f=filename,
            #     input_names=["texts", "text_lengths"],
            #     output_names=["waveform"],
            #     dynamic_axes={
            #         "texts": {1: "num_token"},
            #         "waveform": {0: "num_samples"},
            #     },
            # )

            # exported_program = torch.onnx.export(
            #     model,
            #     inputs,
            #     opset_version=14,
            #     f=filename,
            #     input_names=["texts", "text_lengths", "alignment"],
            #     output_names=["waveform"],
            #     dynamo=True,
            #     dynamic_shapes = (
            #         (1, Dim("tokens", max=1000)),
            #         (1,),
            #         (1, Dim("tokens", max=1000), Dim("time", max=80*80)),
            #     ),
            #     # dynamic_shapes={
            #     #     "texts": (1, Dim.AUTO),
            #     #     "text_lengths": (1,),
            #     #     "alignment": (1, Dim.AUTO, Dim.AUTO),
            #     # },
            #     report=True,
            # )

            # exported_program = torch.export.export(
            #     model,
            #     inputs,
            #     dynamic_shapes=(
            #         (1, Dim("tokens", max=1000)),
            #         (1,),
            #         (1, Dim("tokens", max=1000), Dim("time", max=80*80)),
            #         # (1, Dim.AUTO),
            #         # (1,),
            #         # (1, Dim.AUTO, Dim.AUTO),
            #     ),
            #     # report=True,
            # )

            # sample = exported_program.module().forward(texts, text_lengths, alignment)
            # sample = sample.cpu().numpy()
            # from scipy.io.wavfile import write
            # import numpy as np
            # sample = np.multiply(sample, 32768).astype(np.int16)
            # write("sample_torch.wav", 24000, sample)

            # onnx_program = torch.onnx.export(
            #     exported_program,
            #     inputs,
            #     opset_version=21,
            #     f=filename,
            #     input_names=["texts", "text_lengths", "alignment"],
            #     output_names=["waveform"],
            #     dynamo=True,
            #     optimize=False,
            #     dynamic_shapes={
            #         (1, Dim("tokens", max=1000)),
            #         (1,),
            #         (1, Dim("tokens", max=1000), Dim("time", max=80*80)),
            #         # "texts": (1, Dim.AUTO),
            #         # "text_lengths": (1,),
            #         # "alignment": (1, Dim.AUTO, Dim.AUTO),
            #     },
            #     # report=True,
            # )
            # onnx_program.save(filename)
    # add_meta_data_onnx(filename, "model_config", model_config.model_dump_json())
    return filename
