import re, sys
import inflect
from nltk.tokenize import PunktTokenizer

sent_detector = PunktTokenizer()
inflect_engine = inflect.engine()


def number_to_words(number):
    return re.sub(r",-", "", inflect_engine.number_to_words(number, zero="oh"))


REMOVE_LIST = [
    (re.compile(r"\s"), " "),
    # Remove citations, footnotes, and parenthetical dates
    (re.compile(r"[\[\({](?:.{0,15})[0-9](?:.{0,15})[\]\)}]"), " "),
    (re.compile(r"[\\>\[\]*_/@#⁠﻿↩]"), " "),
    (re.compile(r"[™•]"), ""),
    (re.compile(r"[ʿʾ]"), "’"),
    (re.compile(r"(\W)['‘’](\W)"), r" \1 \2 "),
    (re.compile(r"^['‘’](\W)"), r" \1 "),
    (re.compile(r"(\W)['‘’]$"), r" \1 "),
]

CONVERT_BEFORE = [
    (re.compile(r"[↗↘]"), r" \0 "),
    (re.compile(r"\.\.\."), r" … "),
    (re.compile(r"\.…"), r"."),
    (re.compile(r"%"), r" percent "),
    (re.compile(r"×"), r" times "),
    (re.compile(r"="), r" equals "),
    (re.compile(r"\+"), r" plus "),
    (re.compile(r"&"), r" and "),
    (re.compile(r"°"), r" degrees "),
    (re.compile(r"′"), r" minutes "),
    (re.compile(r"″"), r" seconds "),
    (re.compile(r"℅"), r" care of "),
    (re.compile(r"(—)([‘’'])"), r"\1 \2"),
    (re.compile(r"---*"), r" — "),
    (re.compile(r"\*\*\*+"), r" — "),
    (re.compile(r"\s-\s"), r" — "),
    (re.compile(r"[–⸺⸻]"), r" — "),
    (re.compile(r"(—\.)|(\.—)"), r" . "),
]


def ifnotnone(s):
    result = ""
    if s is not None:
        result = s
    return result


CONVERT_EARLY_NUMBERS = [
    # Year-like but just before a plural noun
    (
        re.compile(r"\b([1-9]\d\d(?:\d?))(?:\s+)([a-rt-z]+s)\b"),
        lambda m: (number_to_words(m.group(1)) + " " + m.group(2)),
    ),
    # Dollars
    (
        re.compile(
            r"(?:\bUS)?\$(\d+)(,[\d,]+)?(\.\d+)?( (?:thousand|million|billion|trillion))?\b"
        ),
        lambda m: (
            number_to_words(m.group(1) + ifnotnone(m.group(2)) + ifnotnone(m.group(3)))
            + ifnotnone(m.group(4))
            + " dollars "
        ),
    ),
    (re.compile(r"\$"), " "),
    # Pounds
    (
        re.compile(
            r"£(\d+)(,[\d,]+)?(\.\d+)?( (?:thousand|million|billion|trillion))?\b"
        ),
        lambda m: (
            number_to_words(m.group(1) + ifnotnone(m.group(2)) + ifnotnone(m.group(3)))
            + ifnotnone(m.group(4))
            + " pounds "
        ),
    ),
    (re.compile(r"£"), " "),
    # Fractions
    (re.compile(r"\b([1-9]\d*)\s*¼"), r" \1 and a quarter "),
    (re.compile(r"\b([1-9]\d*)\s*½"), r" \1 and a half "),
    (re.compile(r"\b([1-9]]d*)\s*¾"), r" \1 and three quarters "),
    (re.compile(r"¼"), " one quarter "),
    (re.compile(r"½"), " one half "),
    (re.compile(r"¾"), " three quarters "),
    # Big Numbers
    (re.compile(r"\b\d+,([\d,]+)(\.\d+)?\b"), lambda m: number_to_words(m.group(0))),
    # Ordinals
    (
        re.compile(r"\b\d+((?:th)|(?:nd)|(?:st)|(?:rd))\b"),
        lambda m: number_to_words(m.group(0)),
    ),
]

CONVERT_YEARS = [
    (re.compile(r"\b(\d?\d)00\'?[sS]\b"), r"\1 hundreds"),
    (re.compile(r"\b(\d?\d)10\'?[sS]\b"), r"\1 tens"),
    (re.compile(r"\b(\d?\d)20\'?[sS]\b"), r"\1 twenties"),
    (re.compile(r"\b(\d?\d)30\'?[sS]\b"), r"\1 thirties"),
    (re.compile(r"\b(\d?\d)40\'?[sS]\b"), r"\1 forties"),
    (re.compile(r"\b(\d?\d)50\'?[sS]\b"), r"\1 fifties"),
    (re.compile(r"\b(\d?\d)60\'?[sS]\b"), r"\1 sixties"),
    (re.compile(r"\b(\d?\d)70\'?[sS]\b"), r"\1 seventies"),
    (re.compile(r"\b(\d?\d)80\'?[sS]\b"), r"\1 eighties"),
    (re.compile(r"\b(\d?\d)90\'?[sS]\b"), r"\1 nineties"),
    (re.compile(r"\b(\d)([1-9]\d)\b"), r"\1 \2"),
    (re.compile(r"\b(\d)0([1-9])\b"), r"\1 oh \2"),
    (re.compile(r"\b(\d\d)([1-9]\d)\b"), r"\1 \2"),
    (re.compile(r"\b(\d\d)0([1-9])\b"), r"\1 oh \2"),
    (re.compile(r"\b(\d\d)00\b"), r"\1 hundred"),
]

CONVERT_NUMBERS = [
    # Time
    (re.compile(r"\b((?:1[0-2])|[1-9]):00\b"), r" \1 o'clock "),
    (re.compile(r"\b((?:1[0-2])|[1-9]):0([1-9])\b"), r" \1 oh \2 "),
    (re.compile(r"\b((?:1[0-2])|[1-9]):([1-5]\d)\b"), r" \1 \2 "),
    # Numbers
    (re.compile(r"\b\d+(,[\d,]+)?(\.\d+)?\b"), lambda m: number_to_words(m.group(0))),
]

# starting quotes
STARTING_QUOTES = [
    (re.compile(r"^\""), r" “ "),
    (re.compile(r"(``)"), r" “ "),
    (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 “ "),
]

# punctuation
PUNCTUATION = [
    (re.compile(r"([:,])([^\d])"), r" \1 \2"),
    (re.compile(r"([:,])$"), r" \1 "),
    (re.compile(r"[;]"), r" \g<0> "),
    (re.compile(r"([^\.])(\.)\s*(”)"), r"\1 \2 \3 "),
    (
        re.compile(r'([^\.])(\.)([\]\)}>"”\']*)\s*$'),
        r"\1 \2\3 ",
    ),  # Handles the final period.
    (re.compile(r"[?!]"), r" \g<0> "),
    (re.compile(r"([^'])' "), r"\1 "),
]

# Pads parentheses
PARENS_BRACKETS = (re.compile(r"[\(\)]"), r" \g<0> ")

# ending quotes
ENDING_QUOTES = [
    (re.compile(r"''"), " ” "),
    (re.compile(r'"'), " ” "),
]

CONVERT_AFTER = [
    (re.compile(r"[‘’]"), "'"),
    (re.compile(r"[“]"), " “ "),
    (re.compile(r"[”]"), " ” "),
    (re.compile(r"\b[-‑]\b"), r" "),
    (re.compile(r"\b[-—]"), r" — "),
    (re.compile(r"[-—]\b"), r" — "),
]


def word_tokenize(text):
    text = text.lower()

    for regexp, substitution in REMOVE_LIST:
        text = regexp.sub(substitution, text)

    for regexp, substitution in CONVERT_BEFORE:
        text = regexp.sub(substitution, text)

    for regexp, substitution in CONVERT_EARLY_NUMBERS:
        text = regexp.sub(substitution, text)

    for regexp, substitution in CONVERT_YEARS:
        text = regexp.sub(substitution, text)

    for regexp, substitution in CONVERT_NUMBERS:
        text = regexp.sub(substitution, text)

    for regexp, substitution in STARTING_QUOTES:
        text = regexp.sub(substitution, text)

    for regexp, substitution in PUNCTUATION:
        text = regexp.sub(substitution, text)

    # Handles parentheses.
    regexp, substitution = PARENS_BRACKETS
    text = regexp.sub(substitution, text)

    # add extra space to make things easier
    text = " " + text + " "

    for regexp, substitution in ENDING_QUOTES:
        text = regexp.sub(substitution, text)

    for regexp, substitution in CONVERT_AFTER:
        text = regexp.sub(substitution, text)

    return text.strip().split()


remove_newlines = re.compile(r"\s")


def sent_tokenize(text):
    return sent_detector.tokenize(remove_newlines.sub(" ", text))


def tokenize(text):
    result = []
    for sentence in sent_tokenize(text):
        result.extend(word_tokenize(sentence))
    return result


ignore_list = {
    "cc0": True,
    "1.a": True,
    "1.e.8": True,
    "1.c": True,
    "1.d": True,
    "1.e": True,
    "1.e.1": True,
    "1.e.2": True,
    "1.e.7": True,
    "1.e.9": True,
    "1.e.3": True,
    "1.e.4": True,
    "1.e.5": True,
    "1.e.6": True,
    "1.f": True,
    "1.f.1": True,
    "1.f.2": True,
    "1.f.3": True,
    "1.f.4": True,
    "1.f.5": True,
    "1.f.6": True,
    "license,1": True,
    "1.b": True,
}

# After tokenization, we expect the following punctuation:
GOOD_SYMBOL = re.compile(r"^[,.;:?!()“”—…↗↘]$")
# GOOD_WORD = re.compile(r"^[\-.'a-zA-Zôèâçéàêæœöîäëøïî']+$")
GOOD_WORD = re.compile(r"^[\-.'a-zA-Z]+$")


def check_tokens(tokens):
    bad_tokens = []
    for i in range(len(tokens)):
        if (
            re.search(GOOD_SYMBOL, tokens[i]) is None
            and re.search(GOOD_WORD, tokens[i]) is None
            and tokens[i] not in ignore_list
        ):
            bad_tokens.append(tokens[i])
    return bad_tokens


# If invoked directly:
#   Read from stdin
#   Tokenize all input
#   Print out any nonconforming tokens
if __name__ == "__main__":
    tokens = tokenize(" ".join(sys.stdin.readlines()))
    sys.stdout.write(" ".join(tokens))
    bad_tokens = check_tokens(tokens)
    if len(bad_tokens) > 0:
        sys.stderr.write("BAD TOKENS: " + " ".join(bad_tokens) + "\n")
