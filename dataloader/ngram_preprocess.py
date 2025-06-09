import json
import numpy as np
import re
import torch

from dataloader.get_book_text import read_austen_chapter, find_start, find_end


def clean_and_tokenise(text: str) -> list:
    """
    Lowercase, strip out punctuation (but keep internal apostrophes),
    collapse whitespace, and split into tokens.
    """
    text = text.lower()
    # replace everything except letters/digits/underscore/space/apostrophe with space
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    # 3) remove any apostrophes that are not between two alphanumeric chars
    #text = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", text)
    # collapse multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()
    # split on whitespace
    return text.split()


def create_vocab(tokens: list) -> list:
    """
    Create a vocabulary from the list of tokens.
    Rank the vocabulary alphabetically
    """
    vocab = sorted(list(set(tokens)))  # Create a sorted list of unique tokens
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def save_vocab(word2idx: dict, idx2word: dict, vocab_path: str = 'austen_no_p&p') -> None:
    """
    Save the vocabulary mappings to a JSON file.
    """
    vocab_data = {'word2idx': word2idx, 'idx2word': idx2word}
    with open(f'vocab_dicts/{vocab_path}.json', 'w') as f:
        json.dump(vocab_data, f)
    print(f"Vocabulary saved to 'vocab_dicts/{vocab_path}.json'")


def load_vocab(vocab_path: str = 'austen_no_p&p') -> tuple:
    """
    Load the vocabulary mappings from a JSON file.
    """
    with open(f'vocab_dicts/{vocab_path}.json', 'r') as f:
        vocab_data = json.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']
    # Convert idx2word keys to integers
    idx2word = {int(k): v for k, v in idx2word.items()}
    return word2idx, idx2word


def tokenise_w_ids(tokens: list, word2idx: dict) -> list:
    """
    Map the tokens to their corresponding indices in the vocabulary.
    """
    tokenised_id_text = [word2idx.get(token, -1) for token in tokens]  # Use -1 for unknown tokens
    return tokenised_id_text


def create_ngrams(token_ids: list, n: int = 2) -> np.ndarray:
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


def prepare_ngram_dataset(text: str, n: int = 2, word2idx: dict = None) -> tuple:
    """
    Prepare a n-gram features and labels from the input text.
    """
    tokens = clean_and_tokenise(text)
    if word2idx is not None:
        token_ids = tokenise_w_ids(tokens, word2idx)
    else:
        word2idx, idx2word = create_vocab(tokens)
        save_vocab(word2idx, idx2word, vocab_path='vocab_{n}_gram')
        token_ids = tokenise_w_ids(tokens, word2idx)

    ngrams = create_ngrams(token_ids, n=n)

    features = ngrams[:,:-1]  # All but the last n-gram
    targets = ngrams[:,-1]  # Predict the next word given the previous n-1 words
    return features, targets


def split_text_train_test_eval(text: str, train_ratio: float = 0.8, test_ratio: float = 0.1, path:str = 'persuasion') -> tuple:
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
    
    train_text = ' '.join(tokenised_text[:train_end])
    test_text = ' '.join(tokenised_text[train_end:test_end])
    eval_text = ' '.join(tokenised_text[test_end:])

    # Save the splits to dict in file
    with open(f'data/split_texts_{path}.json', 'w') as f:
        json.dump({
            'train': train_text,
            'test': test_text,
            'eval': eval_text
        }, f, indent=4)

    return train_text, test_text, eval_text


def split_and_tokenise_text(text: str, n: int = 2, word2idx: dict = None) -> tuple:
    """
    Split the text into training, testing, and evaluation sets,
    then prepare the n-gram dataset for each split.
    """
    train_text, test_text, eval_text = split_text_train_test_eval(text)
    
    train_features, train_targets = prepare_ngram_dataset(train_text, n=n, word2idx=word2idx)
    test_features, test_targets = prepare_ngram_dataset(test_text, n=n, word2idx=word2idx)
    eval_features, eval_targets = prepare_ngram_dataset(eval_text, n=n, word2idx=word2idx)

    # Save the splits to dict in file
    np.savez_compressed(
        f'data/split_ngram_data_{n}_gram.npz',
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
        eval_features=eval_features,
        eval_targets=eval_targets
    )

    return #(train_features, train_targets), (test_features, test_targets), (eval_features, eval_targets)


def main():
    # Create global austen vocab - need for training across all texts
    with open('data/jane_austen_complete.txt', 'r', encoding='utf-8') as file:
        full_text = file.read()
    start_index = find_start('PERSUASION', text=full_text)
    end_index = find_end('PERSUASION', text=full_text)
    persuasion = full_text[start_index:end_index]  # Start from the beginning of PERSUASION
    tokens = clean_and_tokenise(persuasion)
    word2idx, idx2word = create_vocab(tokens)
    save_vocab(word2idx, idx2word, vocab_path='persuasion')
    # Prepare n-gram dataset for PERSUASION
    n = 5
    split_and_tokenise_text(persuasion, n=n, word2idx=word2idx)

if __name__ == "__main__":
    main()