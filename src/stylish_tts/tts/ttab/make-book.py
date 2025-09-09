import argparse, math, os, pathlib, re, subprocess, sys, wave
import numpy as np
import scipy.io.wavfile
import inference
import prepare_book


class BookMetadata:
    def __init__(self):
        self.title = None
        self.author = None
        self.image = None
        self.text = None

    def load_metadata(self, inpath, temppath):
        command = ["ebook-meta", str(inpath)]
        p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        p.wait()
        metadata = p.stdout.readlines()
        for line in metadata:
            if line.startswith("Title"):
                index = line.find(":")
                title = ""
                if index != -1:
                    title = line[index + 1 :]
                if len(title.strip()) > 0:
                    self.title = title.strip()
            elif line.startswith("Author"):
                index = line.find(":")
                author = ""
                if index != -1:
                    author = line[index + 1 :]
                author = re.sub(r"\[[^\]]*\]", r"", author)
                if len(author.strip()) > 0:
                    self.author = author.strip()

    def load_image(self, inpath, temppath):
        imagepath = temppath / "image.jpg"
        command = ["ebook-meta", str(inpath), "--get-cover=" + str(imagepath)]
        p = subprocess.Popen(command)
        p.wait()
        if p.returncode == 0:
            self.image = imagepath

    def load_text(self, path):
        with path.open("r") as f:
            self.text = "\n".join(f.readlines())


def load_book(inpath, temppath):
    result = BookMetadata()
    if inpath.suffix == ".txt" or inpath.suffix == ".md":
        result.load_text(inpath)
    elif (
        inpath.suffix == ".epub" or inpath.suffix == ".azw3" or inpath.suffix == ".mobi"
    ):
        command = [
            "ebook-convert",
            str(inpath),
            str(temppath / "book.txt"),
            "--txt-output-formatting=markdown",
            "--transform-html-rules=calibre-tts-look-and-feel.txt",
        ]
        p = subprocess.Popen(command)
        p.wait()
        result.load_text(temppath / "book.txt")
        result.load_metadata(inpath, temppath)
        result.load_image(inpath, temppath)
    return result


def make_time(num):
    seconds = num // 24000
    minutes = seconds // 60
    hours = minutes // 60
    remainder = (num % 24000) // 24
    seconds = seconds % 60
    minutes = minutes % 60
    return "%02d:%02d:%02d.%03d" % (hours, minutes, seconds, remainder)


def generate_opus_command(path, book_data, marks):
    result = ["/usr/bin/opusenc", "--ignorelength"]
    if book_data.title is not None:
        result.append("--title=" + book_data.title)
    if book_data.author is not None:
        result.append("--artist=" + book_data.author)
    if book_data.image is not None:
        result.append("--picture=" + str(book_data.image))
    for i in range(len(marks)):
        result.append("--comment=CHAPTER%03d=%s" % (i, make_time(marks[i][1])))
        result.append("--comment=CHAPTER%03dNAME=%s" % (i, marks[i][0]))
    result += ["-", str(path)]
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="../../models/bc-phon-09.yml")
parser.add_argument("--model", default="../../models/bc-phon-09.pth")
parser.add_argument("--tmp", default="../../audio")
parser.add_argument("--styledir", default="../../StyleTTS2")
parser.add_argument("infile")
parser.add_argument("outfile")
args = parser.parse_args()

inpath = pathlib.Path(args.infile)
outpath = pathlib.Path(args.outfile)
temppath = pathlib.Path(args.tmp)

os.system("mkdir -p " + str(temppath / "wav"))

book_data = load_book(inpath, temppath)
markdown = re.sub(r"(\s)(#+)\s+(\S)", r"\1\2 \3", book_data.text)
chapters = prepare_book.prepare(markdown)

current = 0
current_time = 0
total = 0
for ch in chapters:
    total += len(ch[1].split("\n"))

model = inference.Model(args.styledir, args.config, args.model)
chapter_marks = []

for ch in chapters:
    sys.stdout.write("\n" + ch[0] + "\n")
    chapter_marks.append((ch[0], current_time))
    texts = ch[1].strip().split("\n")
    for i in range(len(texts)):
        ps = texts[i].strip()
        if current % 50 == 0:
            sys.stdout.write("\n%d/%d\n" % (current, total))
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        next_text = None
        if i < len(texts) - 1:
            next_text = texts[i + 1].strip()
        audio = model.inference(
            ps, diffusion_steps=7, embedding_scale=1, next_text=next_text
        )
        current_time += audio.shape[0]
        temp_wav = str(temppath / "temp.wav")
        audio = np.multiply(audio, 32768).astype(np.int16)
        scipy.io.wavfile.write(temp_wav, 24000, audio)
        out_wav = str(temppath / "wav" / ("%08d.wav" % current))
        pad = ""
        if "↘" in ps:
            pad = " pad 0.5 0.5 "
            current_time += 24000
        os.system("sox -G %s %s norm %s" % (temp_wav, out_wav, pad))
        current += 1

total = current
total_time = current_time
command = generate_opus_command(outpath, book_data, chapter_marks)
print(command)
process = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
out_wav = None
for i in range(total):
    in_wav_name = str(temppath / "wav" / ("%08d.wav" % i))
    with wave.open(in_wav_name, mode="rb") as w:
        if out_wav is None:
            out_wav = wave.open(process.stdin, mode="wb")
            out_wav.setparams(w.getparams())
        out_wav.writeframesraw(w.readframes(w.getnframes()))
process.stdin.close()
try:
    out_wav.close()
except:
    pass
process.wait()
