import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class EnsembleNGramModel(nn.Module):
    """
    A model that ensembles multiple pretrained n-gram models by learning scalar weights.
    The ensemble logits are a weighted sum of individual model logits, where weights are learned.
    """
    def __init__(self, ngram_models: list[nn.Module]):
        super().__init__()
        self.ngram_models = nn.ModuleList(ngram_models)
        self.num_models = len(ngram_models)
        # initialize ensemble weights equally
        init_w = torch.ones(self.num_models) / self.num_models
        self.weights = nn.Parameter(init_w)  # shape: (num_models,)

    def forward(self, contexts: list[torch.LongTensor]) -> torch.Tensor:
        """
        contexts: list of LongTensors, each of shape (batch_size, context_size_i)
                  contexts[i] corresponds to the i-th ngram model's required context size.
        Returns:
            ensemble_logits: Tensor of shape (batch_size, vocab_size)
        """
        logits_list = []
        # collect logits from each ngram model
        for model, ctx in zip(self.ngram_models, contexts):
            model.eval()
            logits = model(ctx)       # shape: (batch_size, vocab_size)
            logits_list.append(logits)
        # stacked: (num_models, batch_size, vocab_size)
        stacked = torch.stack(logits_list, dim=0)
        # apply weights (broadcast over batch & vocab)
        weighted = stacked * self.weights.view(self.num_models, 1, 1)
        ensemble_logits = weighted.sum(dim=0)
        return ensemble_logits


class NGramEnsembleDataset(Dataset):
    """
    Dataset that for each target token returns the contexts for each ngram model
    and the target index.
    """
    def __init__(self, token_ids: list[int], ngram_models: list[nn.Module]):
        self.ngram_models = ngram_models
        self.n_values = [m.context_size for m in ngram_models]
        self.vocab_size = ngram_models[0].embeddings.num_embeddings
        # build examples: slide a window of length = max(n_values)+1
        max_ctx = max(self.n_values)
        self.examples = []  # list of (list of contexts, target)
        for i in range(len(token_ids) - max_ctx):
            # for each model, extract its context of size n-1
            contexts = []
            for ctx_size in self.n_values:
                start = i + max_ctx - ctx_size
                end = start + ctx_size
                contexts.append(torch.tensor(token_ids[start:end], dtype=torch.long))
            target = token_ids[i + max_ctx]
            self.examples.append((contexts, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        contexts, target = self.examples[idx]
        return contexts, torch.tensor(target, dtype=torch.long)


