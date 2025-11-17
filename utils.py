import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    #raise NotImplementedError

    import random
    from nltk.corpus import wordnet
    from nltk import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    random.seed(0)
    detok = TreebankWordDetokenizer()

    # Nearby keyboard letters for typo simulation
    qwerty_neighbors = {
        "a": ["s", "q", "z"], "s": ["a", "d", "x"],
        "e": ["w", "r", "d"], "i": ["u", "o", "k"],
        "o": ["i", "p", "l"], "t": ["r", "y", "g"],
        "n": ["b", "m", "h"], "r": ["e", "t", "f"]
    }

    def add_typo(word):
        """Randomly replace one character with a nearby key."""
        if len(word) < 4:
            return word
        idx = random.randint(0, len(word) - 1)
        ch = word[idx].lower()
        if ch in qwerty_neighbors:
            new_char = random.choice(qwerty_neighbors[ch])
            word = word[:idx] + new_char + word[idx + 1:]
        return word

    def get_synonym(word):
        """Return a random synonym if available."""
        syns = [
            lemma.name().replace("_", " ")
            for syn in wordnet.synsets(word)
            for lemma in syn.lemmas()
            if lemma.name().lower() != word.lower()
        ]
        return random.choice(syns) if syns else word

    # --- main transformation ---
    words = word_tokenize(example["text"])
    new_words = []

    for w in words:
        # (1) randomly delete short/common words
        if len(w) <= 3 and random.random() < 0.3:
            continue

        # (2) replace ~40% of eligible words with synonyms
        if w.isalpha() and len(w) > 3 and random.random() < 0.4:
            w = get_synonym(w)

        # (3) inject typos into ~25% of words
        elif w.isalpha() and random.random() < 0.25:
            w = add_typo(w)

        # (4) insert random punctuation into ~5% of words
        elif random.random() < 0.05:
            w = w + random.choice([",", ".", "!", "??"])

        new_words.append(w)

    example["text"] = detok.detokenize(new_words)

    ##### YOUR CODE ENDS HERE ######

    return example
