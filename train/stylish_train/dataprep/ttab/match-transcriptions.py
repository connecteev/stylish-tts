import os, sys, re
from difflib import SequenceMatcher
import phonemizer
import argparse
import pathlib

# from nltk import tokenize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ttab import tokens

espeak = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=False,
    with_stress=False,
    words_mismatch="ignore",
)


def is_junk(word):
    return word.strip("1234567890,.;:-?!'\"()$%—“”‘’") == ""


class Chapter:
    def __init__(self, filename):
        self.book = None
        self.next_start = 0
        self.matcher = SequenceMatcher(autojunk=False)

        self.book = tokenize_book(filename)
        self.matcher.set_seq1(self.book.gt_clean)

    def filterjunk(self, words):
        result = []
        for word in words:
            if not is_junk(word):
                result.append(word)
        return result

    def match_next(self, text, begin_time, end_time):
        words = tokens.word_tokenize(text)
        phonemes = espeak.phonemize(words)
        filtered = self.filterjunk(phonemes)
        self.matcher.set_seq2(filtered)
        match = self.matcher.find_longest_match(self.next_start)
        if match.size < len(filtered) or match.size == 0:
            print("skipped|%s|%s" % (begin_time, end_time))
            # print("partial|" + " ".join(self.book.gt_base[self.book.clean_to_start[match.a]:self.book.clean_to_end[match.a + match.size]]) + "|" + " ".join(words))
        else:
            phrase = " ".join(self.grow_tokens(match.a, match.a + match.size))
            print("phrase|%s|%s|%s" % (begin_time, end_time, phrase.strip()))
            self.next_start = match.a + match.size

    def grow_tokens(self, begin, end):
        done = False
        begin = self.book.clean_to_start[begin]
        end = self.book.clean_to_end[end]
        while begin > 0 and not done:
            candidate = self.book.gt_base[begin - 1]
            if candidate in "\"'(“‘":
                begin = begin - 1
            else:
                done = True
        done = False
        while end < len(self.book.gt_base) and not done:
            candidate = self.book.gt_base[end]
            if candidate in "\"'),.;:-?!”’":
                end = end + 1
            else:
                done = True
        return self.book.gt_base[begin:end]


book_mapping = {}


def get_chapter(base, chapterfile):
    global book_mapping
    result = None
    filename = None
    for key in book_mapping.keys():
        if key in chapterfile:
            filename = book_mapping[key]
            break
    if filename is not None:
        result = Chapter(base / filename)
    return result


book_cache = {}


class BookTokens:
    def __init__(self, filename):
        self.gt_base = []
        self.gt_clean = []
        self.clean_to_start = []
        self.clean_to_end = []

        with filename.open(mode="r") as f:
            text = " ".join(f.readlines())
            self.gt_base = tokens.tokenize(text)
        start_index = 0
        last_nonjunk = 0
        ph = espeak.phonemize(self.gt_base)
        for word in ph:
            if not is_junk(word):
                self.gt_clean.append(word)
                self.clean_to_start.append(start_index)
                self.clean_to_end.append(last_nonjunk + 1)
                last_nonjunk = start_index
            start_index += 1
        self.clean_to_start.append(start_index)
        self.clean_to_end.append(last_nonjunk + 1)


def tokenize_book(filename):
    if str(filename) in book_cache:
        result = book_cache[str(filename)]
    else:
        result = BookTokens(filename)
        book_cache[str(filename)] = result
    return result


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--base", default=".")
args = arg_parser.parse_args()
base = pathlib.Path(args.base)

# Initialize book_mapping
with (base / "raw/book-mapping.txt").open(mode="r") as f:
    for line in f:
        fields = line.split("|")
        if len(fields) > 1:
            book_mapping[fields[0].strip()] = fields[1].strip()
chapter = None
for line in sys.stdin:
    fields = line.split("|")
    if fields[0] == "chapter":
        sys.stderr.write("\nChapter: " + fields[1].strip())
        sys.stderr.flush()
        chapter = get_chapter(base, fields[1].strip())
        print(line.strip())
    elif fields[0] == "skipped":
        print(line.strip())
    elif fields[0] == "phrase":
        if chapter is not None:
            sys.stderr.write(".")
            sys.stderr.flush()
            chapter.match_next(fields[3].strip(), fields[1].strip(), fields[2].strip())
        else:
            print("skipped|%s|%s" % (fields[1].strip(), fields[2].strip()))
    sys.stdout.flush()
