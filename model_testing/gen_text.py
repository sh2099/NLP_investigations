import json
import torch


def load_model_and_vocab(model_class: torch.nn.Module, n: int, model_path: str, vocab_path: str) -> tuple:
    """
    Load the trained model and vocabulary from the specified paths.
    """
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']
    # Convert idx2word keys to integers
    idx2word = {int(k): v for k, v in idx2word.items()}
    # Initialise the model
    model = model_class(vocab_size=len(word2idx), n=n)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model, word2idx, idx2word


def generate_text_n_gram(model: torch.nn.Module, 
                         prompt: str, 
                         word2idx: dict, 
                         idx2word:dict, 
                         max_length: int = 50
                         ) -> None:
    print(f"Prompt: {prompt}")
    start_words = prompt.split()  # Split the prompt into words
    model.eval()
    generated_text = start_words.copy()
    k = model.context_size  
    for _ in range(max_length - len(start_words)):
        # Prepare input tensor from the last n words
        #TODO: Add padding if the length of generated_text is less than n
        # Probably a good idea to add padding to vocab
        input_tensor = torch.tensor([word2idx[word] for word in generated_text[-k:]], dtype=torch.long)
        #print(input_tensor, input_tensor.shape)
            # Change this into logging
        #Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        #print(input_tensor, input_tensor.shape)
            # Again logging maybe
        # Get model predictions
        with torch.no_grad():
            predictions = model(input_tensor)
            _, pred_label = torch.max(predictions, dim=1)
            #pred_label = torch.argmax(predictions, dim=1).item()
            generated_text.append(idx2word[pred_label.item()])

    print(' '.join(generated_text))


def main():
    # Load the model and vocabulary
    from ml.n_gram_model import NGramModel
    n = 5
    model_path = f'trained_models/{n}-gram_model_persuasion.pth'
    vocab_path = f'vocab_dicts/persuasion.json'
    model, word2idx, idx2word = load_model_and_vocab(NGramModel, n, model_path, vocab_path)
    #print(idx2word[2292])

    # Example usage
    start_prompt = 'he turned over the' 
    start_words = start_prompt  # Split the prompt into words
    max_length = 300  # Generate up to 50 words

    generate_text_n_gram(model, start_words, word2idx, idx2word, max_length=max_length)

if __name__ == "__main__":
    main()