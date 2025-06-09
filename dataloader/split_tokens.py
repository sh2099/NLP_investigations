import json
import numpy as np
from pathlib import Path

from dataloader.create_ngrams import prepare_ngram_dataset
from dataloader.tokenise import clean_and_tokenise
from dataloader.vocab import Vocab


def split_text_train_test_eval(text: str, 
                               train_ratio: float = 0.8, 
                               test_ratio: float = 0.1, 
                               path:str = 'persuasion'
                               ) -> tuple[str, str, str]:
    """
    Split the text into training, testing, and evaluation sets.
    """
    eval_ratio = 1 - train_ratio - test_ratio
    if eval_ratio < 0:
        raise ValueError("Train ratio + Test ratio must be less than or equal to 1.")
    tokenised_text = clean_and_tokenise(text)
    length = len(tokenised_text)
    train_end = int(length * train_ratio)
    test_end = int(length * (train_ratio + test_ratio))
    
    splits = {
        'train' : ' '.join(tokenised_text[:train_end]),
        'test' : ' '.join(tokenised_text[train_end:test_end]),
        'eval' : ' '.join(tokenised_text[test_end:]),
    }

    with open(f'data/split_texts_{path}.json', 'w') as f:
        json.dump(splits, f, indent=2)

    return splits['train'], splits['test'], splits['eval']


def split_and_tokenise_text(
    text: str,
    n: int = 2,
    vocab: Vocab | None = None,
    vocab_path: Path | str = None
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    Vocab
]:
    """
    1) split text → train/test/eval strings,
    2) for each split call prepare_ngram_dataset,
    3) save compressed .npz of all three splits,
    4) return the three (features, targets) and final vocab.
    """
    train_txt, test_txt, eval_txt = split_text_train_test_eval(text, path=vocab_path)

    train_X, train_y, vocab = prepare_ngram_dataset(train_txt, n, vocab, vocab_path)
    test_X,  test_y,  _     = prepare_ngram_dataset(test_txt,  n, vocab, None)
    eval_X,  eval_y,  _     = prepare_ngram_dataset(eval_txt,  n, vocab, None)

    # save everything to a compressed .npz
    np.savez_compressed(
        f'data/split_{vocab_path}_data_{n}_gram.npz',
        train_X=train_X, train_y=train_y,
        test_X=test_X,   test_y=test_y,
        eval_X=eval_X,   eval_y=eval_y
    )

    return (train_X, train_y), (test_X, test_y), (eval_X, eval_y), vocab
