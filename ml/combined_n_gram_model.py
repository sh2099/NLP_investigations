import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from ml.n_gram_model import NGramModel
from ml.ensemble_ngram_model import EnsembleNGramModel



def load_n_gram_model(n: int = 4, work: str = 'austen_no_p&p', word2idx: dict = None) -> tuple:
    """
    Load the n-gram model and vocabulary from the specified paths.
    """
    model_path = f'trained_models/{n}-gram_model_{work}.pth'
    # Initialise the model
    model = NGramModel(vocab_size=len(word2idx), n=n)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return model


def load_n_gram_models(n_range: range = range(2, 11), work: str = 'austen_no_p&p', word2idx: dict = None) -> list:
    """
    Load a list of n-gram models for the specified range of n.
    """
    models = []
    for n in n_range:
        model = load_n_gram_model(n, work, word2idx)
        models.append(model)
    return models


def load_vocab(vocab_path: str = 'austen_full_vocab') -> tuple:
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


def train_meta_model(
    ensemble_model: EnsembleNGramModel,
    dataset: Dataset,
    batch_size: int = 64,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu")
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    ensemble_model.to(device)
    ensemble_model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for contexts, targets in dataloader:
            # move to device
            contexts = [c.to(device) for c in contexts]
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = ensemble_model(contexts)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return ensemble_model


def _collate_fn(batch):
    """
    Collate a batch of (contexts_list, target) into batched contexts and targets.
    """
    # batch is list of tuples: (list_of_context_tensors, target_tensor)
    num_models = len(batch[0][0])
    # for each model, stack its contexts
    batched_contexts = []
    for model_idx in range(num_models):
        ct = torch.stack([examples[0][model_idx] for examples in batch], dim=0)
        batched_contexts.append(ct)
    targets = torch.stack([t for _, t in batch], dim=0).squeeze()
    return batched_contexts, targets

# Example usage:
# 1. Load pretrained n-gram models (order 2 to 10) trained on 90% Austen except P&P
#    ngram_models = [load_model(n) for n in range(2, 11)]
# 2. Prepare token_ids for the held-out 10% data
#    heldout_ids = tokenize_and_encode(heldout_text)
# 3. Create ensemble_model = MetaEnsembleNGramModel(ngram_models)
# 4. dataset = NGramEnsembleDataset(heldout_ids, ngram_models)
# 5. trained_ensemble = train_meta_model(ensemble_model, dataset)
# 6. To fine-tune on P&P, repeat steps 2-5 with P&P splits, starting from trained_ensemble.