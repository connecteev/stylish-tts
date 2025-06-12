from scipy.io.wavfile import write
import inference
import argparse, os, sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="result.wav")
parser.add_argument("--model", default="/home/duerig/proj/tts/models/bc-phon-06.pth")
parser.add_argument("--config", default="/home/duerig/proj/tts/models/bc-phon-06.yml")
parser.add_argument("--styledir", default="../../StyleTTS2/")

# parser.add_argument("--quiet", default=False)
args = parser.parse_args()

model = inference.Model(args.styledir, args.config, args.model)

results = []
lines = sys.stdin.readlines()
for i in range(len(lines)):
    line = lines[i]
    next_line = lines[0].strip()
    next_line = None
    if i < len(lines) - 1:
        next_line = lines[i + 1].strip()
    audio = model.inference(
        line.strip(), diffusion_steps=7, embedding_scale=1, next_text=next_line
    )
    # audio = model.context_infer(
    #    i, lines, diffusion_steps=7, embedding_scale=1)
    results.append(audio)
    sys.stderr.write(".")
    sys.stderr.flush()

sys.stderr.write("\n")
sys.stderr.flush()
combined = np.multiply(np.concatenate(results), 32768)
print("Saving to:", args.out)
write(args.out, 24000, combined.astype(np.int16))

os.system(
    "~/proj/tts/ffmpeg/bin/ffplay.exe -autoexit -nodisp -hide_banner -loglevel quiet %s"
    % args.out
)
