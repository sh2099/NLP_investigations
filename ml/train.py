import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dataloader.get_book_text import read_austen_chapter,read_austen_chapters, read_austen_work
from dataloader.ngram_preprocess import prepare_ngram_dataset, save_vocab, load_vocab
from model_testing.gen_text import generate_text_n_gram
from ml.n_gram_model import NGramModel


def create_dataloader(features: list, targets: list, batch_size: int = 640) -> DataLoader:
    """
    Create a DataLoader for the features and targets.
    """
    dataset = TensorDataset(torch.tensor(features, dtype=torch.long), torch.tensor(targets, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(
        model: nn.Module, 
        dataloader: DataLoader, 
        num_epochs:int = 50, 
        learning_rate: int = 0.01
        ) -> None:
    """
    Train the n-gram model using the provided DataLoader.
    """
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for features, targets in dataloader:  
            optimiser.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        total_loss /= len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')



def train_and_save(model: nn.Module, 
        dataloader: DataLoader, 
        num_epochs: int = 50, 
        learning_rate: float = 0.01, 
        n: int = 4,
        work: str = 'PERSUASION'
    ) -> None:
    """
    Train the n-gram model and save it to a file.
    """
    model_path = f'trained_models/{n}-gram_model_{work.lower()}.pth'
    train_model(model, dataloader, num_epochs, learning_rate)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")



def load_chapters_and_train_ngram(work: str, 
                                  chapter_range: tuple = (1, 5), 
                                  n: int = 4, 
                                  num_epochs: int = 50,
                                  prompt: str = 'he turned over the',
                                  ) -> tuple:
    """
    Load chapters from a work and prepare the n-gram dataset.
    """
    # === Load chapters and create mappings ===
    text = read_austen_chapters(work, chapter_range)
    word2idx, idx2word = load_vocab(f'{work.lower()}')  # Load existing vocab or create new one
    features, targets,  = prepare_ngram_dataset(text, n=n, word2idx=word2idx)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")
    save_vocab(word2idx, idx2word, vocab_path=f'{work.lower()}')

    # === Create DataLoader and initalise Model ===
    dataloader = create_dataloader(features, targets)
    model = NGramModel(vocab_size=vocab_size, n=n)
    print(model)

    # === Train the model and generate text ===
    print("Generating text with untrained model:")
    generate_text_n_gram(model, prompt, word2idx, idx2word, max_length=10)
    train_and_save(model, dataloader,num_epochs=num_epochs, n=n, work=work)
    print("Generating text with trained model:")
    generate_text_n_gram(model, prompt, word2idx, idx2word, max_length=10)

def main():
    work_name = 'PERSUASION'
    chapter_range = (1, 24)
    n = 5
    num_epochs = 100
    prompt = 'he turned over the'
    
    # Load chapters and train the n-gram model
    load_chapters_and_train_ngram(work_name, chapter_range, n,num_epochs, prompt)

if __name__ == "__main__":
    main()