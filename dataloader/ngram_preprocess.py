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

text = read_austen_chapter('PERSUASION', 1)
print("Original Text:", text[:100])
tokens = clean_and_tokenise(text)
print("Cleaned Tokens:", tokens[:10])