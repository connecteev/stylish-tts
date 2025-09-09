import phonemizer
import re, sys
import json
import ttab.homographs
import ttab.tokens
import ttab.data
import importlib.resources

# from transformers import T5ForConditionalGeneration, AutoTokenizer

# char_model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_small_100')
# char_model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
# char_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

# def char_phoneme(word):
#    # tokenized English words
#    wordlist = ['<eng-us>: ' + word]
#
#    out = char_tokenizer(wordlist, padding = True, add_special_tokens = False, return_tensors = 'pt')
#    preds = char_model.generate(**out, num_beams = 1, max_length = 50)
#    return char_tokenizer.batch_decode(preds.tolist(), skip_special_tokens = True)[0]

TO_ESPEAK = [
    (re.compile(r"ɔt"), r"ɔːt"),
    (re.compile(r"ɔɹ"), r"ɔːɹ"),
    (re.compile(r"ɨd"), r"ᵻd"),
    (re.compile(r"ɝˈ"), r"ˈɝ"),
    (re.compile(r"ɫ"), r"l"),
    (re.compile(r"i([^ː])"), r"iː\1"),
    (re.compile(r"ɑ([^ː])"), r"ɑː\1"),
    (re.compile(r"u([^ː])"), r"uː\1"),
    (re.compile(r"ɝ([^ː])"), r"ɜː\1"),
    (re.compile(r"i$"), r"iː"),
    (re.compile(r"ɑ$"), r"ɑː"),
    (re.compile(r"u$"), r"uː"),
    (re.compile(r"ɝ$"), r"ɜː"),
    (re.compile(r"ɨ"), r"ɪ"),
    (re.compile(r"˨"), r""),
    (re.compile(r"ʧ"), r"tʃ"),
    (re.compile(r"ʤ"), r"dʒ"),
    (re.compile(r"\u035C"), r""),
    (re.compile(r"\u0361"), r""),
    (re.compile(r"\u203F"), r""),
    (re.compile(r"\u032F"), r""),
    (
        re.compile(
            r"([ˈˌ])([^iyɪeʏøɛæœaɨɘʉəɜɵɐäɞʊɯɤʌɑuoɔɒː]+)([iyɪeʏøɛæœaɨɘʉəɜɵɐäɞʊɯɤʌɑuoɔɒː])"
        ),
        r"\2\1\3",
    ),
]


def to_espeak(word):
    result = word
    for regexp, substitution in TO_ESPEAK:
        result = regexp.sub(substitution, result)
    return result


def is_punctuation(word):
    return len(word) == 1 and word in ",.;:?!()“”—…"


def pluralize(phonemes):
    if phonemes[-1] in "szʃʒ":
        return phonemes + "əz"
    elif phonemes[-1] in "iyɪeʏøɛæœaɨɘʉəɜɵɐäɞʊɯɤʌɑuoɔɒː":
        return phonemes + "z"
    else:
        return phonemes + "s"


class Lexicon:
    def __init__(self, source):
        self.source = source
        self.children = {}
        self.fallback = None

    def set_fallback(self, phonemes, source):
        if self.fallback is None:
            self.fallback = phonemes
            self.source = source

    def add_item(self, source, path):
        word, phonemes = path[0]
        if word not in self.children:
            self.children[word] = Lexicon(source)
        if len(path) == 1:
            self.children[word].set_fallback(phonemes, source)
        else:
            self.children[word].add_item(source, path[1:])

    def lookup(self, path):
        if len(path) == 0:
            return self.fallback, self.source, path
        elif path[0] not in self.children:
            return self.fallback, self.source, path
        else:
            return self.children[path[0]].lookup(path[1:])


class Phoneme:
    def __init__(self):
        self.hl = ttab.homographs.HomographLexicon()
        self.hl.load()
        self.lex = Lexicon(".")
        self.espeak = phonemizer.backend.EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
            words_mismatch="ignore",
        )
        self.cache = {}

        self.load_lexicon("ttab.lexicon", "T")
        self.load_lexicon("cmu.lexicon", "C")
        self.load_lexicon("mfa.lexicon", "M")

    def load_lexicon(self, filename, source):
        with importlib.resources.open_text(ttab.data, filename) as f:
            for line in f:
                fields = line.split("|")
                key = fields[0].strip()
                if (
                    " " not in key
                    and key[0] != "-"
                    and key[-1] != "-"
                    and not key.isdigit()
                ):
                    value = fields[1].strip()
                    wordlist = key.split("-")
                    path = []
                    for word in wordlist:
                        path.append((word, None))
                    path = path[:-1] + [(wordlist[-1], value)]
                    self.lex.add_item(source, path)

    def lookup(self, wordlist):
        if len(wordlist) == 0:
            return None, []
        # First try a multi-word recursive lookup
        phonemes, source, path = self.lex.lookup(wordlist)
        word = wordlist[0]
        # Try removing periods for acronym finding
        if phonemes is None:
            word = word.replace(".", "")
            phonemes, source, _ = self.lex.lookup([word])
            path = wordlist[1:]
        # Try removing single quotes
        if phonemes is None:
            word = word.strip("'")
            phonemes, source, _ = self.lex.lookup([word])
            path = wordlist[1:]
        # Try nonplural variant
        if phonemes is None:
            word = re.sub(r"'?s", "", wordlist[0])
            phonemes, source, _ = self.lex.lookup([word])
            path = wordlist[1:]
            if phonemes is not None:
                phonemes = pluralize(phonemes)
        if phonemes is None:
            return None, ".", wordlist
        else:
            return phonemes, source, path

    def pronounce_sentence(self, text):
        result = []
        wordlist = ttab.tokens.word_tokenize(text)
        index = 0
        path = wordlist
        while len(path) > 0:
            word = path[0]
            if is_punctuation(word):
                result.append(word)
                index += 1
                path = path[1:]
            elif self.hl.has(word):
                # Homograph
                homograph = self.hl.pronounce(index, wordlist)
                result.append(to_espeak(homograph))
                index += 1
                path = path[1:]
            else:
                phonemes, _, nextpath = self.lookup(path)
                if phonemes is None:
                    if word in self.cache:
                        phonemes = self.cache[word]
                    else:
                        phonemes = self.espeak.phonemize([word])[0]
                        self.cache[word] = phonemes
                    result.append(phonemes)
                    index += 1
                    path = path[1:]
                else:
                    result.append(to_espeak(phonemes))
                    index += len(path) - len(nextpath)
                    path = nextpath

        return " ".join(result)

    def check_sentence(self, sentence, already={}):
        wordlist = ttab.tokens.word_tokenize(sentence)
        path = wordlist
        while len(path) > 0:
            if is_punctuation(path[0]) or path[0] in already:
                path = path[1:]
            else:
                phonemes, source, nextpath = self.lookup(path)
                if phonemes is None:
                    already[path[0]] = True
                    sys.stdout.write(
                        "UNKNOWN WORD: [" + path[0] + "]" + " ".join(wordlist) + "\n\n"
                    )
                    sys.stdout.flush()
                    # sys.stderr.write(".")
                    path = path[1:]
                else:
                    # sys.stderr.write(source)
                    path = nextpath
                # sys.stderr.flush()
        # sys.stderr.write("\n")
        # sys.stderr.flush()


# If invoked directly:
#   Read from stdin,
#   Print out any failed lookups to stdout,
#   Print source lexicon for all lookups to stderr
if __name__ == "__main__":
    already = {}
    p = Phoneme()
    # p.lookup("eager's")
    lines = sys.stdin.readlines()
    text = " ".join(lines)
    sentences = ttab.tokens.sent_tokenize(text)
    for sentence in sentences:
        p.check_sentence(sentence, already=already)
