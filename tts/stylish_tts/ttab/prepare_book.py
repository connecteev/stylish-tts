# Split markdown text book into chapters and phonemize it

import json, sys, io, re
import mistletoe
from mistletoe.ast_renderer import AstRenderer
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ttab.phonemes
import ttab.tokens

TEXT = 0
HEADER = 1
phonemizer = ttab.phonemes.Phoneme()


class Book:
    def __init__(self):
        self.chapters = [[]]
        self.new_section(TEXT)

    def write(self, text):
        self.chapters[-1][-1][0].write(text)

    def new_chapter(self):
        self.chapters.append([])

    def new_section(self, marker):
        self.chapters[-1].append((io.StringIO(""), marker))

    def phonemize(self):
        self.filter_empty()
        result = []
        i = 1
        for chapter in self.chapters:
            title = "Chapter " + str(i)
            if chapter[0][1] == HEADER:
                title = chapter[0][0].getvalue().strip()
            phonemes = phonemize_chapter(chapter)
            result.append((title, phonemes))
            # if i > 3:
            #    break
            i += 1
        return result

    def filter_empty(self):
        chapters = []
        for ch in self.chapters:
            newchapter = []
            for section in ch:
                if len(section[0].getvalue().strip()) > 0:
                    newchapter.append(section)
            if len(newchapter) > 0:
                chapters.append(newchapter)
        self.chapters = chapters


def print_blocks(f, node_list):
    in_quote = False
    for node in node_list:
        if node["type"] == "Quote" and not in_quote:
            in_quote = True
            f.write("Quote.\n")
        elif node["type"] != "Quote" and in_quote:
            in_quote = False
            f.write("Unquote.\n\n")
        if node["type"] == "Heading":
            print_heading(f, node, node["children"])
        elif node["type"] == "Paragraph":
            print_spans(f, node["children"])
        elif node["type"] == "List":
            print_list(f, node["children"])
        elif node["type"] == "Quote":
            print_blocks(f, node["children"])
        else:
            raise Exception("Unknown block: " + node["type"])
        f.write("\n\n")


def print_spans(f, node_list):
    for node in node_list:
        if node["type"] == "RawText":
            f.write(node["content"])
        elif node["type"] == "EscapeSequence":
            print_escape(f, node["children"])
        elif node["type"] == "Emphasis":
            print_spans(f, node["children"])
        elif node["type"] == "LineBreak":
            f.write("\n")
        elif node["type"] == "Strong":
            print_spans(f, node["children"])
        else:
            raise Exception("Unknown Span: " + node["type"])


def print_heading(f, parent, node_list):
    if parent["level"] == 1:
        # print("NEW CHAPTER", node_list)
        f.new_chapter()
    f.new_section(HEADER)
    print_spans(f, node_list)
    f.new_section(TEXT)


def print_escape(f, node_list):
    for node in node_list:
        if node["type"] == "RawText":
            f.write(node["content"])
        else:
            raise Exception("Unknown Span: " + node["type"])


def print_list(f, node_list):
    f.write("List.\n")
    for node in node_list:
        if node["type"] == "ListItem":
            f.write("Item.\n")
            print_blocks(f, node["children"])
        else:
            raise Exception("Unknown Span: " + node["type"])
    f.write("End List.\n")


roman_table = {
    "i": "first",
    "ii": "second",
    "iii": "third",
    "iv": "fourth",
    "v": "fifth",
    "vi": "sixth",
    "vii": "seventh",
    "viii": "eighth",
    "ix": "ninth",
    "x": "tenth",
    "xi": "eleventh",
    "xii": "twelvth",
    "xiii": "thirteenth",
    "xiv": "fourteenth",
    "xv": "fifteenth",
    "xvi": "sixteenth",
    "xvii": "seventeenth",
    "xviii": "eighteenth",
    "xix": "nineteenth",
    "xx": "twentieth",
}


def royalty_sub(match):
    ordinal = None
    roman = match.group(2).lower()
    if roman in roman_table:
        return match.group(1) + " the " + roman_table[roman]
    else:
        return match.group(0)


def fix_royalty(book):
    return re.sub(r"\b([A-Z]\w+)\s([xXvViI]+)\b", royalty_sub, book)


def find_split(ps):
    found = None
    for i in range(300):
        if ps[i] in ",.;:?!—…":
            found = i + 1
    if found is None:
        for i in range(300, 500):
            if ps[i] == " ":
                found = i + 1
                break
    if found is None:
        found = 500
    return found


def force_smaller(ps):
    result = []
    while len(ps) > 500:
        index = find_split(ps)
        result.append(ps[:index])
        ps = ps[index:]
    result.append(ps)
    return result


def phonemize_chapter(chapter):
    result = []
    for section in chapter:
        text = section[0].getvalue().strip()
        text = fix_royalty(text)
        text = phonemize_section(text)
        if section[1] == HEADER:
            text = " ↗ " + text.strip() + " ↘ "
        result.append(text)
    return "\n".join(result)


def phonemize_section(section):
    texts = []
    if len(section) > 100:
        texts = ttab.tokens.sent_tokenize(section)
    else:
        texts = [section.strip()]
    segments = []
    current = 1
    for t in texts:
        current += 1
        if current % 50 == 0:
            sys.stderr.write("\n%d/%d\n" % (current, len(texts)))
        else:
            sys.stderr.write(".")
            sys.stderr.flush()
        ps = phonemizer.pronounce_sentence(t)
        segments.extend(force_smaller(ps))
    result = ""
    linelen = 0
    for s in segments:
        if linelen > 200 or linelen + len(s) > 500:
            result += "\n"
            linelen = 0
        else:
            result += " "
        result += s
        linelen += len(s)
    # If the final line is too short, try to merge and resplit it
    # lines = result.strip().split("\n")
    # if len(lines) > 1 and len(lines[-1]) < 100:
    #    remerged = force_smaller(" ".join(lines[-2:-1]))
    #    result = "\n".join(lines[:-2] + remerged)
    return result.strip()


def prepare(bookfile):
    jsonstring = mistletoe.markdown(bookfile, AstRenderer)
    ast = json.loads(jsonstring)
    book = Book()
    print_blocks(book, ast["children"])
    return book.phonemize()
