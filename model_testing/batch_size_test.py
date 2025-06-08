import torch

from ml.n_gram_model import NGramModel
from ml.train import create_dataloader, train_and_save
from dataloader.ngram_preprocess import prepare_ngram_dataset
from dataloader.get_book_text import read_austen_chapters
from model_testing.gen_text import generate_text_n_gram


def train_model_batch_size(features: list, targets: list, vocab_size, batch_size: int, n: int = 4) -> None:
    """
    Evaluate the model with a specific batch size.
    """
    # Create dataloader for the specified batch size
    dataloader = create_dataloader(features, targets, batch_size=batch_size)  
    model = NGramModel(vocab_size, n=n)
    # Train and save the model
    train_and_save(model, dataloader, n=n, model_path=f'trained_models/{n}-gram_model_{batch_size}.pth')


def gen_batch_models(n: int = 4, batch_sizes: list = None) -> None:
    """
    Generate and train models for different batch sizes.
    """
    if batch_sizes is None:
        batch_sizes = [32, 64, 128, 256, 512, 1024]
    # Load the text data
    text = read_austen_chapters('PERSUASION', (1, 5))
    features, targets, word2idx, idx2word = prepare_ngram_dataset(text, n=n)
    vocab_size = len(word2idx)
    for batch_size in batch_sizes:
        print(f"Training model with batch size: {batch_size}")
        train_model_batch_size(features, targets, vocab_size, batch_size, n=n)
        print(f"Model trained and saved for batch size: {batch_size}\n")



if __name__ == "__main__":
    # Uncomment to generate and train models for different batch sizes
    gen_batch_models(n=4, batch_sizes=[32, 64, 128])
