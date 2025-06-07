import torch
import torch.nn as nn
import torch.nn.functional as F


class NGramModel(nn.Module):
    """
    An n-gram language model using PyTorch.

    This model takes the previous (n-1) tokens and predicts the next (nth) token.

    Args:
        vocab_size (int): Number of unique tokens in the vocabulary.
        embedding_dim (int): Dimensionality of token embeddings.
        context_size (int): Number of context tokens (default: 3 for 4-gram).
        hidden_dim (int): Number of units in the hidden linear layer.
    """
    def __init__(self, vocab_size: int, embedding_dim: int=64, n: int = 3, hidden_dim: int = 128) :
        super(NGramModel, self).__init__()
        self.n = n
        self.context_size = n-1
        # A simple feed-forward network: embed -> hidden -> output
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear((n - 1) * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.LongTensor) -> torch.Tensor:
        # Get embeddings: [batch_size, context_size, embedding_dim]
        embeds = self.embeddings(inputs)
        # Flatten context embeddings: [batch_size, context_size * embedding_dim]
        concat = embeds.view(inputs.size(0), -1)
        hidden = F.relu(self.linear1(concat))
        logits = self.linear2(hidden)
        logits = F.log_softmax(logits, dim=1)  # Apply log softmax to get log probabilities
        return logits
