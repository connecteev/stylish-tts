import sys

import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ttab.phonemes
import ttab.tokens

p = ttab.phonemes.Phoneme()
text = " ".join(sys.stdin.readlines())
sentences = ttab.tokens.sent_tokenize(text)
for sentence in sentences:
    print(p.pronounce_sentence(sentence))
