import numpy as np
from pathlib import Path

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
