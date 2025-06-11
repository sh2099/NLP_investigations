from dataloader.get_book_text import find_start, find_end
from dataloader.split_tokens import split_text_train_test_eval
from dataloader.tokenise import clean_and_tokenise
from dataloader.vocab import Vocab


def main():
    # 1) read full Austen text, extract "Emma"
    with open('data/jane_austen_complete.txt', 'r', encoding='utf-8') as f:
        full = f.read()
    start = find_start('PERSUASION', text=full)
    end   = find_end(  'PERSUASION', text=full)
    persuasion = full[start:end]
    del full

    # 2) build & save vocab on Persuasion
    tokens = clean_and_tokenise(persuasion)
    vocab  = Vocab(tokens)
    vocab.save('persuasion')

    # 3) split into train/test/eval, save splits to json
    train_txt, test_txt, eval_txt = split_text_train_test_eval(persuasion, path='persuasion')

if __name__ == "__main__":
    main()