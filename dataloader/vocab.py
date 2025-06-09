import json
from pathlib import Path
from typing import List


class Vocab:
    def __init__(self, tokens: List[str]):
        self.word2idx, self.idx2word = self._build(tokens)

    @staticmethod
    def _build(tokens):
        vocab = sorted(list(set(tokens)))
        w2i = {w:i for i,w in enumerate(vocab)}
        i2w = {i:w for w,i in w2i.items()}
        return w2i, i2w

    def save(self, path: Path):
        vocab_data = {'word2idx': self.word2idx, 'idx2word': self.idx2word}
        vocab_path = f'vocab_dicts/{path}.json'
        with open(vocab_path, 'w') as f:
            # Save the vocabulary data as JSON
            json.dump(vocab_data, f, indent=2)
        #print(f"Vocabulary saved to 'vocab_dicts/{vocab_path}.json'")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        data = json.load(path.open('r'))
        v = cls.__new__(cls)
        v.word2idx = data['word2idx']
        v.idx2word = {int(k):v for k,v in data['idx2word'].items()}
        return v

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.word2idx.get(t, self.word2idx.get("<unk>")) for t in tokens]
