import torchtext
import re

glove = torchtext.vocab.GloVe(name="840B", dim=300)   # embedding size = 300

for (i,j) in torchtext.vocab:
    print(i)
    break