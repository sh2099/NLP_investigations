import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
from dataloader.get_chapter import read_austen_chapter
from dataloader.ngram_preprocess import prepare_ngram_dataset

torch.manual_seed(0)  # For reproducibility

class BigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super(BigramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Embed the input into a continuous space
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        # Want to map back to vocabulary
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)  # Convert input indices to embeddings
        x = self.linear(x)
        return x


def train_model(model, features, targets, num_epochs=50, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        optimiser.zero_grad()
        outputs = model(features)
        # CrossEntropyLoss expects raw logits, not probabilities
    
        loss = criterion(outputs, targets)
        loss.backward()
        optimiser.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    print(f'Accuracy: {accuracy:.4f}')


def generate_text(model, start_words, word2idx, idx2word, n=2, max_length=50):
    print(f"Prompt: {' '.join(start_words)}")
    model.eval()
    generated_text = start_words.copy()
    for _ in range(max_length - len(start_words)):
        # Prepare input tensor from the last n words
        #TODO: Add padding if the length of generated_text is less than n
        # Probably a good idea to add padding to vocab
        input_tensor = torch.tensor([word2idx[word] for word in generated_text[-n+1:]], dtype=torch.long)

        # Add batch dimension
        #input_tensor = input_tensor.unsqueeze(0)

        # Get model predictions
        with torch.no_grad():
            predictions = model(input_tensor)
            #print(f"Predictions: {predictions}, len: {len(predictions)}")
            pred_label = torch.argmax(predictions, dim=1).item()
            generated_text.append(idx2word[pred_label])

    print(' '.join(generated_text))



def main():
    # Example usage
    work_name = 'PERSUASION'
    chapter_number = 1
    print(f'Reading chapter {chapter_number} of {work_name}')
    chapter_text = read_austen_chapter(work_name, chapter_number)
    
    # Prepare the n-gram dataset
    n = 2  # Bigram
    print(f'Preparing {n}-gram dataset')
    features, targets, word2idx, idx2word = prepare_ngram_dataset(chapter_text, n=n)
    print(features.reshape(-1).shape, targets.shape)
    vocab_size = len(word2idx)

    print(f"Initializing Bigram model with vocabulary size: {vocab_size}")
    model = BigramModel(vocab_size=vocab_size)
    print(model)
    
    # Make some pre-training predictions
    print("Generating text with untrained model:")
     # Example: Generate text starting with 'it is'
    start_words = ['there', 'was']  # Example starting words
    generate_text(model, start_words, word2idx, idx2word, n=n, max_length=10)

    # Train the model
    train_model(model, features.reshape(-1), targets, num_epochs=50)
    
    # Now that the model is trained, we can generate text
    print("Generating text with trained model:")
    generate_text(model, start_words, word2idx, idx2word, n=n, max_length=10)


if __name__ == "__main__":
    main()