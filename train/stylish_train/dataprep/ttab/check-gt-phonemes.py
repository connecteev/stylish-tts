import sys
from ttab.lib.phonemes import Phoneme

p = Phoneme()
already = {}

for line in sys.stdin:
    fields = line.split("|")
    if fields[0] == "phrase":
        p.check_sentence(fields[3].strip(), already=already)
