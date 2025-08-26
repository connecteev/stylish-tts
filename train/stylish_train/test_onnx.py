import torch
from stylish_lib.config_loader import ModelConfig, load_model_config_yaml
from stylish_lib.text_utils import TextCleaner
import torch
import numpy as np
import torch
import onnxruntime as ort
import click
from scipy.io.wavfile import write
import onnx
from time import perf_counter
from utils import DurationProcessor
import ai_edge_torch


def read_meta_data_onnx(filename, key):
    model = onnx.load(filename)
    for prop in model.metadata_props:
        if prop.key == key:
            return prop.value
    return None


@click.command()
@click.option("--stylish_path", type=str)
@click.option("--duration_path", type=str)
@click.option("--model_config_path", type=str)
@click.option("--text", type=str, help="A list of phonemes")
@click.option("--combine", type=bool, default=True, help="Combine to one file")
def main(stylish_path, duration_path, model_config_path, text, combine):
    texts = [text]
    # model_config = read_meta_data_onnx(stylish_path, "model_config")
    # assert (
    #     model_config
    # ), "model_config metadata not found. Please rerun ONNX conversion."
    # model_config = ModelConfig.model_validate_json(model_config)
    model_config = load_model_config_yaml(model_config_path)
    text_cleaner = TextCleaner(model_config.symbol)
    dur_session = ai_edge_torch.load(duration_path)
    # dur_session = ort.InferenceSession(
    #     dur_path,
    #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    # )
    duration_processor = DurationProcessor(16, 50)
    # session = ort.InferenceSession(
    #     stylish_path,
    #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    # )
    session = ai_edge_torch.load(stylish_path)
    samples = []

    start = perf_counter()
    for i, text in enumerate(texts):
        tokens = torch.tensor(text_cleaner(text)).unsqueeze(0)
        texts = torch.zeros([1, tokens.shape[1] + 2], dtype=int)
        texts[0][1 : tokens.shape[1] + 1] = tokens
        text_lengths = torch.zeros([1], dtype=int)
        text_lengths[0] = tokens.shape[1] + 2
        text_mask = torch.zeros(1, texts.shape[1], dtype=bool)
        # Load ONNX model
        # dur_pred = dur_session.run(
        #     None,
        #     {
        #         "texts": texts.cpu().numpy(),
        #         "text_lengths": text_lengths.cpu().numpy(),
        #     },
        # )
        dur_pred = dur_session(texts.cpu().numpy(), text_lengths.cpu().numpy())

        dur_pred = torch.Tensor(dur_pred).squeeze(0)
        alignment = duration_processor(dur_pred, text_lengths).unsqueeze(0)

        # outputs = session.run(
        #     None,
        #     {
        #         "texts": texts.cpu().numpy(),
        #         "text_lengths": text_lengths.cpu().numpy(),
        #         "alignment": alignment.cpu().numpy(),
        #     },
        # )
        outputs = session(
            texts.cpu().numpy(), text_lengths.cpu().numpy(), alignment.cpu().numpy()
        )
        samples.append(np.multiply(outputs, 32768).astype(np.int16))

    if combine:
        outfile = "sample_combined.wav"
        combined = np.concatenate(samples, axis=-1)
        print("Saving to:", outfile)
        write(outfile, 24000, combined)
    else:
        for i, sample in enumerate(samples):
            outfile = f"sample_{i}.wav"
            print("Saving to:", outfile)
            write(outfile, 24000, sample)
    time = perf_counter() - start
    print(f"{time}s, RTF {time/(combined.shape[0] / 24000)}")


if __name__ == "__main__":
    main()
