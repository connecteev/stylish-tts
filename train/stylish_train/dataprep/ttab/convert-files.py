import os, pathlib, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ttab import phonemes

phon = phonemes.Phoneme()

base = pathlib.Path("/mnt/z/fine/Gemini-2.0-Flash-Kore-Voice/Kore")
names = []
with open("/mnt/z/fine/wavlist.txt", "r") as f:
    for line in f.readlines():
        names.append(line.strip()[:-4])

for name in names:
    textpath = base / (name + ".txt")
    with textpath.open("r") as f:
        text = " ".join(f.readlines()).strip()
        ps = phon.pronounce_sentence(text)
        print("%s.wav|%s|0" % (name, ps))
