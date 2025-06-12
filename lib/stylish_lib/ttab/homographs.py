from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch, numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sys, math, json, re
import ttab.tokens
import ttab.data
import importlib.resources
import nltk
import spacy

spacy_engine = spacy.load("en_core_web_trf")

device = "cpu"


class HomographLexicon:
    def __init__(self):
        model_id = "answerdotai/ModernBERT-large"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(model_id)
        self.homographs = {}
        self.homograph_classes = {}
        self.class_phonemes = {}
        self.homograph_pos = {}
        self.good = 0
        self.bad = 0
        self.model_cache = {}

    def load(self):
        self.load_vectors()
        self.load_classes()
        self.load_lexicon()
        self.load_pos()

    def load_vectors(self, filename=None):
        if filename is None:
            f = importlib.resources.open_binary(
                ttab.data, "homograph-training-vectors.npz"
            )
        else:
            f = open(filename, "rb")
        with f as v:
            vectors = numpy.load(v, allow_pickle=False)
            for key in vectors.keys():
                vlist = vectors.get(key)
                if len(vlist.shape) == 0:
                    continue
                self.homographs[key] = []
                for i in range(vlist.shape[0]):
                    self.homographs[key].append(vlist[i])
            vectors.close()

    def load_classes(self, filename=None):
        if filename is None:
            f = importlib.resources.open_text(
                ttab.data, "homograph-training-classes.json"
            )
        else:
            f = open(filename, "r")
        with f as v:
            self.homograph_classes = json.load(v)

    def load_lexicon(self):
        with importlib.resources.open_text(ttab.data, "homographs.lexicon") as f:
            for line in f:
                fields = line.split("|")
                self.class_phonemes[fields[0]] = fields[1].strip()

    def load_pos(self):
        with importlib.resources.open_text(ttab.data, "homograph-pos.json") as f:
            self.homograph_pos = json.load(f)

    # For saving homograph training data
    def save_vectors(self, filename):
        numpy.savez_compressed(filename, allow_pickle=False, **(self.homographs))

    def save_classes(self, filename):
        with open(filename, "w") as f:
            json.dump(self.homograph_classes, f)

    # Generate homograph training data from datasets
    def generate_homographs(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                values = line.split("\t")
                word = values[0].strip('"')
                print(".", end="")
                sys.stdout.flush()
                if word == "homograph":
                    continue
                if word not in self.homographs:
                    self.homographs[word] = []
                    self.homograph_classes[word] = []
                vector = self.get_sense_vector(
                    values[2].strip('"'), int(values[3]), int(values[4])
                )
                self.homographs[word].append(vector.detach().numpy())
                self.homograph_classes[word].append(values[1].strip('"'))

    # Test homograph training data on a validation dataset
    def test_homographs(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                values = line.split("\t")
                word = values[0].strip('"')
                if word == "homograph":
                    continue

                text = values[2].strip('"')
                m = re.search(r"\b" + word + r"\b", text.lower())
                if m is not None:
                    begin = m.start()
                    end = m.end()
                    correct = values[1].strip('"')
                    answer = self.get_classid(text, begin, end)

                    if answer == correct:
                        self.good += 1
                    else:
                        self.bad += 1
                        print("Failed:", word, answer, correct, values[2].strip('"'))
                        # print("Correct Answers %d/%d" % (self.good,
                        #                                 self.good + self.bad))
                else:
                    print("FAILED TO FIND:" + word)

    # Actually pick a homograph of a word in sentence (wordlist)
    def pick_homograph(self, index, wordlist):
        before = (" ".join(wordlist[:index])) + " "
        word = wordlist[index]
        after = " " + (" ".join(wordlist[index + 1 :]))
        start = len(before)
        end = start + len(word)
        classid = self.get_classid(before + word + after, start, end)
        return classid

    def pronounce(self, index, wordlist):
        choice = self.pick_homograph(index, wordlist)
        return self.class_phonemes[choice]

    def get_classid(self, text, start, end):
        word = text[start:end].lower()
        test_vector = self.get_sense_vector(text, start, end)
        if word in self.model_cache:
            model = self.model_cache[word]
        else:
            model = LogisticRegression(random_state=0, max_iter=1000)
            model.fit(self.homographs[word], self.homograph_classes[word])
            self.model_cache[word] = model
        odds = model.predict_proba([test_vector.detach().numpy()])[0]
        confident = False
        for o in odds:
            if o > 0.9:
                confident = True

        result = None
        if not confident:
            # Backup is to try to do part of speech matching
            pos_choice = None
            doc = spacy_engine(text)
            wordtag = None
            for item in doc:
                if item.idx == start:
                    key = word + "|" + item.tag_
                    wordtag = key
                    if key not in self.homograph_pos:
                        key = word + "|" + item.tag_[:2]
                    if key in self.homograph_pos:
                        result = self.homograph_pos[key]
        if result is None:
            result = model.predict([test_vector.detach().numpy()])[0]
        return result

    def has(self, word):
        return word in self.homographs

    def get_sense_vector(self, text, start, end):
        inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            add_special_tokens=True,
        ).to(device)
        outputs = self.bert_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        state = outputs.hidden_states[-1].squeeze()
        mapping = inputs["offset_mapping"].squeeze()
        index = 1
        while index < len(mapping) - 1 and mapping[index][1] <= start:
            index += 1
        start_index = index
        while index < len(mapping) - 1 and mapping[index][1] <= end:
            index += 1
        end_index = index
        result = torch.zeros(state[1].size())
        divisor = end_index - start_index
        for item in state[start_index:end_index]:
            result += item.to("cpu") / divisor
        return result


if __name__ == "__main__":
    count = 0
    hl = HomographLexicon()
    hl.load()
    text = " ".join(sys.stdin.readlines())
    sentences = ttab.tokens.sent_tokenize(text)
    for sentence in sentences:
        words = ttab.tokens.word_tokenize(sentence)
        for i in range(len(words)):
            if hl.has(words[i]):
                print(
                    words[i] + "|" + hl.pick_homograph(i, words) + "|" + " ".join(words)
                )
            sys.stderr.write(".")
            sys.stderr.flush()
        sys.stderr.write("%d/%d\n" % (count, len(sentences)))
        sys.stderr.flush()
        count += 1
