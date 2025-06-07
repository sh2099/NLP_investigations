import json
import torch

from ml.bigram_model import BigramModel

model_path = 'trained_models/bigram_model.pth'
vocab_path = 'trained_models/bigram_vocab.json'


def load_model_and_vocab(model_path, vocab_path):
    """
    Load the trained model and vocabulary from the specified paths.
    """
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']
    # Conver idx2word keys to integers
    idx2word = {int(k): v for k, v in idx2word.items()}
    # Initialise the model
    model = BigramModel(vocab_size=len(word2idx))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model, word2idx, idx2word


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
    # Load the model and vocabulary
    model, word2idx, idx2word = load_model_and_vocab(model_path, vocab_path)
    #print(idx2word[2292])

    # Example usage
    start_words = ['the', 'man']
    n = 2  # Bigram
    max_length = 50  # Generate up to 50 words

    generate_text(model, start_words, word2idx, idx2word, n=n, max_length=max_length)

if __name__ == "__main__":
    main()