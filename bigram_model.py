import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score


torch.manual_seed(0)  # For reproducibility

class BigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super(BigramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Embed the input into a continuous space
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Want to map back to vocabulary
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)  # Convert input indices to embeddings
        x = x.mean(dim=1) # Average embeddings for bigram input
        x = self.linear(x)
        return x


def train_model(model, data_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in data_loader:
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')