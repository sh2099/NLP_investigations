import numpy as np
import re
import torch

from dataloader.get_chapter import read_austen_chapter


def clean_and_tokenise(text):
    """
    Clean and tokenize the input text.
    This function should be implemented to handle specific cleaning and tokenization tasks.
    """
    # Example implementation (to be replaced with actual logic)
    cleaned_text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    tokens = cleaned_text.split()  # Split by whitespace
    return tokens

def create_vocab(tokens):
    """
    Create a vocabulary from the list of tokens.
    This function should be implemented to handle specific vocabulary creation tasks.
    """
    vocab = sorted(list(set(tokens)))  # Create a sorted list of unique tokens
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def tokenise_w_ids(tokens, word2idx=None):
    """
    Tokenize the input text and return a list of token IDs.
    This function should be implemented to handle specific tokenization tasks.
    """
    tokenised_id_text = [word2idx.get(token, -1) for token in tokens]  # Use -1 for unknown tokens
    return tokenised_id_text

def create_ngram_dataset(token_ids, n=2):
    """
    Create a dataset of bigrams from the list of tokens.
    This function should be implemented to handle specific bigram dataset creation tasks.
    """
    ngrams = []
    for i in range(len(token_ids) - n + 1):
        ngram = token_ids[i:i + n]
        if -1 not in ngram:
            ngrams.append(ngram)
    # Convert to tensor
    #bigrams = torch.tensor(bigrams, dtype=torch.long)
    return np.array(ngrams)


def prepare_ngram_dataset(text, n=2):
    """
    Prepare a dataset of n-grams from the input text.
    This function should be implemented to handle specific n-gram dataset preparation tasks.
    """
    tokens = clean_and_tokenise(text)
    word2idx, idx2word = create_vocab(tokens)
    token_ids = tokenise_w_ids(tokens, word2idx)
    ngrams = create_ngram_dataset(token_ids, n=n)

    features = torch.tensor(ngrams[:,:-1], dtype=torch.long)  # All but the last n-gram
    targets = torch.tensor(ngrams[:,-1], dtype=torch.long)  # Predict the next word given the previous n-1 words
    return features, targets, word2idx, idx2word


def main():
    work_name = 'PERSUASION'
    chapter_number = 1
    chapter_text = read_austen_chapter(work_name, chapter_number)

    # Prepare the n-gram dataset
    n = 2  # Bigram
    features, targets, word2idx, idx2word = prepare_ngram_dataset(chapter_text, n=n)

    print("Features (Bigrams):", features)
    print("Targets (Next words):", targets)
    print("Vocabulary:", word2idx)
    
if __name__ == "__main__":
    main()