import json
import torch

from torch.utils.data import Dataset

from dataloader.tokenise import clean_and_tokenise
from dataloader.vocab import Vocab

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from dataloader.tokenise import clean_and_tokenise
from dataloader.vocab import Vocab

class NGramDataset(Dataset):
    def __init__(self, work: str, text_path: str, n: int = 2, split: str = 'train'):
        """
        Args:
            work:   name of text (e.g. 'persuasion')
            text_path: path to the JSON {"train": "...", "test": "...", "eval": "..."}
            n: the n in n-gram
            split: which split to load: 'train'|'test'|'eval'
        """
        # 1) load vocab
        vocab_file = Path('vocab_dicts') / f"{work.lower()}.json"
        self.vocab = Vocab.load(vocab_file)
        
        # 2) load train/test/eval text
        with open(text_path, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        raw = splits[split]   
        
        # 3) tokenize and encode
        tokens = clean_and_tokenise(raw)
        self.data = self.vocab.encode(tokens)
        
        self.n = n
        self.ctx_size = n - 1

        # 4) Precompute all features & targets -> faster at iteration
        N = len(self.data) - self.ctx_size
        self.features = torch.empty((N, self.ctx_size), dtype=torch.long)
        self.targets  = torch.empty((N,), dtype=torch.long)
        for i in range(N):
            window = self.data[i : i + n]
            # not implemented in earlier code - but keep for later
            if -1 in window:
                window = [w if w>=0 else self.vocab.word2idx["<unk>"] for w in window]
            self.features[i] = torch.tensor(window[:-1])
            self.targets[i]  = window[-1]

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, idx):
        # now just index into your precomputed tensors
        return self.features[idx], self.targets[idx]
