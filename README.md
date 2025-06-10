<div align="center">

# NLP investigations

[![python](https://img.shields.io/badge/-Python_3.13-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
  
</div>

This repository contains the code for a personal project investigating various methods for Natural Language Processing.\
These methods cannot compete with the modern advancements of transformer based methods, but are nevertheless interesting to explore and to test the limits of. 

## N-gram models

Can N-gram models write a Jane Austen Novel?

N-gram models are basic probability based models, which predict the nth token given the previous n-1 tokens. They essentially store probabilities of given n-sequences, which are learned from the frequencies of the sequence occuring in the text the model is trained on. Text is then generated based on the 'most seen' n-sequence.\
These models have a few large issues which prevent them from performing well:
  - No deep understanding of semantic structure - in real text specific words should be paid more attention to, whereas others can be ignored
  - Repetitive and uninteresting text -> since the next word is chosen as the one with the highest probability, certain sequences will always generate the same text
  - Small values of N often lead to looping structure
    - E.g. a -> b -> c -> a
  - Large values of N become too nuanced
    - E.g. most sequences of e.g. 10 words are likely to show up only once in a whole book

The first issue is slighly harder to solve, but here we look and combatting the other problems in the following manner:
  - Introducing probability based sampling to make the text less consistent and less likely to fall into a loop
    - Rather than always taking the highest probability token, draw a token from the token probability distribution
  - Combining multiple levels of N-grams and training a model to learn the weighting of each N-gram by predicting text

For these investigations, the works of Jane Austen are used.\
The N-grams are built based on all of her works available on project Gutenburg, apart from Pride and Prejudice.\
The combined N-gram model is then trained on a subset of the P&P text, and the remainder of the book is used as a test to see how well the text can be reconstructed.


