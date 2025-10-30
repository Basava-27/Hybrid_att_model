import json
from collections import Counter

class CharTokenizer:
    def __init__(self, texts, vocab_size=None):
        counter = Counter("".join(texts))
        if vocab_size:
            counter = dict(counter.most_common(vocab_size))
        self.chars = sorted(counter.keys())
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join([self.itos.get(i, "?") for i in ids])

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls([""])
        tok.stoi, tok.itos = data["stoi"], data["itos"]
        return tok
