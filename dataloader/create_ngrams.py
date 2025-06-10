import json
import numpy as np
from pathlib import Path
import re

from dataloader.get_book_text import find_start, find_end
from dataloader.tokenise import clean_and_tokenise
from dataloader.vocab import Vocab



def create_ngrams(token_ids: list[int], n: int = 2) -> np.ndarray:
    """
    Create a dataset of n-grams from the list of tokens.
    """
    ngrams = []
    # Create n-grams by sliding a window of size n over the token_ids
    for i in range(len(token_ids) - (n - 1)):
        ngram = token_ids[i:i+n]
        if -1 not in ngram:
            ngrams.append(ngram)
    return np.array(ngrams)


def prepare_ngram_dataset(text: str, 
                          n: int = 2, 
                          vocab: Vocab | None = None,
                          vocab_path: Path | str = None,
                          ) -> tuple:
    """
    Tokenize `text`, (re)build or reuse a Vocab, save it if `vocab_path` given,
    then return (features, targets).
    """
    # tokenise the text
    tokens = clean_and_tokenise(text)

    # build or reuse the vocab
    if vocab is None:
        vocab = Vocab(tokens)
    if vocab_path is not None:
        vocab.save(vocab_path)

    # encode to integer IDs (unknown -> -1)
    token_ids = vocab.encode(tokens)

    ngrams = create_ngrams(token_ids, n=n)
    features = ngrams[:,:-1]  # All but the last n-gram
    targets = ngrams[:,-1]  # Predict the next word given the previous n-1 words
    return features, targets, vocab


def main():
    # 1) read full Austen text, extract "Emma"
    with open('data/jane_austen_complete.txt', 'r', encoding='utf-8') as f:
        full = f.read()
    start = find_start('EMMA', text=full)
    end   = find_end(  'EMMA', text=full)
    persuasion = full[start:end]
    del full

    # 2) build & save vocab on Persuasion
    tokens = clean_and_tokenise(persuasion)
    vocab  = Vocab(tokens)
    vocab.save('emma')

    # 3) create & save 5-gram splits (+ get the final vocab object back)
    from dataloader.split_tokens import split_and_tokenise_text
    split_and_tokenise_text(
        text       = persuasion,
        n          = 5,
        vocab      = vocab,
        vocab_path = 'emma'
    )

if __name__ == "__main__":
    main()



#### TODO: Lots of datasets will arise given each split of different n-grams.
# May be better to just save the vocab and the raw text split into train/test/eval.
# Then create the n-grams on-the-fly when training.