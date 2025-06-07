import numpy as np
import re

from dataloader.get_book_text import read_austen_chapter


def clean_and_tokenise(text: str) -> list:
    """
    Clean the text to extract lower case words with no punctuation,
    then split into individual words to act as tokens.
    """
    cleaned_text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    tokens = cleaned_text.split()  # Split by whitespace
    return tokens

def create_vocab(tokens: list) -> list:
    """
    Create a vocabulary from the list of tokens.
    Rank the vocabulary alphabetically
    """
    vocab = sorted(list(set(tokens)))  # Create a sorted list of unique tokens
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
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


def prepare_ngram_dataset(text: str, n: int = 2) -> tuple:
    """
    Prepare a dataset of n-grams from the input text.
    """
    tokens = clean_and_tokenise(text)
    word2idx, idx2word = create_vocab(tokens)
    token_ids = tokenise_w_ids(tokens, word2idx)
    ngrams = create_ngrams(token_ids, n=n)

    features = ngrams[:,:-1]  # All but the last n-gram
    targets = ngrams[:,-1]  # Predict the next word given the previous n-1 words
    return features, targets, word2idx, idx2word


def main():
    work_name = 'PERSUASION'
    chapter_number = 1
    chapter_text = read_austen_chapter(work_name, chapter_number)

    # Prepare the n-gram dataset
    n = 4
    features, targets, word2idx, idx2word = prepare_ngram_dataset(chapter_text, n=n)

    print("Features (N-1)grams):", features[:10])
    print("Targets (Next words):", targets[:10])
    #print("Vocabulary:", word2idx)


if __name__ == "__main__":
    main()