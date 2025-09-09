from pydub import AudioSegment
import random, os, sys, re, pathlib, argparse
from ttab.phonemes import Phoneme


prefix = "micro"

phon = Phoneme()


def save_segment(chapter_index, phrase_index, audio, text):
    global phon, trainfile
    audio_len = len(audio)
    if audio_len <= 1000:
        if len(text) > 0:
            ps = phon.pronounce_sentence(text)
            if len(ps) < 500:
                filename = "%s-%04d-%05d.wav" % (prefix, chapter_index, phrase_index)
                audio.export(
                    str(base / "wav" / filename),
                    format="wav",
                    parameters=["-ar", "24000"],
                )
                f = None
                if random.random() < 0.03:
                    f = valfile
                else:
                    f = trainfile
                f.write("%s|%s|0\n" % (filename, ps))
                f.flush()
                sys.stderr.write(".")
                sys.stderr.flush()
            else:
                print(
                    "Skipping phrase "
                    + str(phrase_index)
                    + " for "
                    + str(len(ps))
                    + " phonemes\n"
                )
        else:
            print("Skipping phrase " + str(phrase_index) + " for empty transcription\n")
    else:
        print("Skipping phrase " + str(phrase_index) + " for " + str(len(audio)) + "\n")


def seek_audio(index, phrases, chapter_length):
    text = ""
    goal = 0.0
    end_index = index
    start = 0
    end = 0
    while index < len(phrases) and phrases[index][2] is None:
        index += 1
    if index < len(phrases):
        count = 0
        start = max(0, phrases[index][0] - 50)
        if index > 0 and phrases[index - 1][1] is not None:
            start = max(phrases[index - 1][1], start)
        end = start
        done = False
        while not done:
            can_lookahead = (
                index < len(phrases) - 1 and phrases[index + 1][2] is not None
            )
            end = min(chapter_length, phrases[index][1] + 50)
            if can_lookahead:
                end = min(phrases[index + 1][0], end)
                if phrases[index + 1][1] - start > 20000:
                    done = True
            else:
                done = True
            length = end - start
            if length > goal:
                done = True
            text = text + " " + phrases[index][2]
            count = count + 1
            index = index + 1
    end_index = index
    return (end_index, start, end, text)


parser = argparse.ArgumentParser()
parser.add_argument("--base", default=".")
args = parser.parse_args()
base = pathlib.Path(args.base)


def main_method():
    chapters = {}
    filename = base / "raw/match-merged.txt"
    with filename.open(mode="r") as f:
        name = ""
        for line in f:
            fields = line.split("|")
            if fields[0] == "chapter":
                name = fields[1].strip()
                chapters[name] = []
            elif fields[0] == "phrase":
                begin = int(fields[1].strip())
                end = int(fields[1].strip())
                chapters[name].append(
                    (int(fields[1].strip()), int(fields[2].strip()), fields[3].strip())
                )
            else:
                chapters[name].append((None, None, None))

    chapter_number = 1
    chapter_total = len(chapters.keys())
    for key in chapters.keys():
        print("(%d/%d) Processing %s\n" % (chapter_number, chapter_total, key))
        chapter_audio = AudioSegment.from_mp3(str(base / key))
        chapter_length = len(chapter_audio)
        phrases = chapters[key]
        index = 0
        while index < len(phrases):
            (index, begin, end, text) = seek_audio(index, phrases, chapter_length)
            if end - begin < 1000:
                save_segment(chapter_number, index, chapter_audio[begin:end], text)
        chapter_number += 1


trainfile = (base / "train-list-micro.txt").open("w", encoding="utf-8")
valfile = (base / "val-list-micro.txt").open("w", encoding="utf-8")
main_method()
trainfile.close()
valfile.close()
