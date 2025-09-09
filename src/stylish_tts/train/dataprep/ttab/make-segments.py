from pydub import AudioSegment
import random, os, sys, re, pathlib, argparse
from ttab.lib.phonemes import Phoneme


prefix = "a"

phon = Phoneme()


def save_segment(chapter_index, phrase_index, audio, text):
    global phon, train400, train800, train1200, train1600
    audio_len = len(audio)
    if audio_len > 1000 and audio_len < 20000:
        if len(text) > 0:
            ps = phon.pronounce_sentence(text)
            if len(ps) < 500:
                filename = "%s-%04d-%05d.wav" % (prefix, chapter_index, phrase_index)
                audio.export(
                    "wav/" + filename, format="wav", parameters=["-ar", "24000"]
                )
                f = None
                if random.random() < 0.03:
                    f = valfile
                else:
                    if audio_len < 5000:
                        f = train400
                    elif audio_len < 10000:
                        f = train800
                    elif audio_len < 15000:
                        f = train1200
                    else:
                        f = train1600
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
    goal = random.gauss(10000, 5000)
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
            if length > 1000 and length > goal:
                done = True
            text = text + " " + phrases[index][2]
            count = count + 1
            index = index + 1
    end_index = index
    return (end_index, start, end, text)


def main_method():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    args = parser.parse_args()
    base = pathlib.Path(args.base)

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
            save_segment(chapter_number, index, chapter_audio[begin:end], text)
        chapter_number += 1


train400 = open("train-list-400.txt", "w", encoding="utf-8")
train800 = open("train-list-800.txt", "w", encoding="utf-8")
train1200 = open("train-list-1200.txt", "w", encoding="utf-8")
train1600 = open("train-list-1600.txt", "w", encoding="utf-8")
valfile = open("val-list.txt", "w", encoding="utf-8")
main_method()
train400.close()
train800.close()
train1200.close()
train1600.close()
valfile.close()
