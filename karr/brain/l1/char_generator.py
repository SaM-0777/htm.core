import os
import string
import random

def simple_lowercase_char_generator(output="dataset/simple_lowercase.txt"):
    chars = string.ascii_lowercase

    with open(output, "w", encoding="utf-8") as f:
        f.write(chars)

def simple_uppercase_char_generator(output="dataset/simple_uppercase.txt"):
    chars = string.ascii_uppercase

    with open(output, "w", encoding="utf-8") as f:
        f.write(chars)

def complex_char_generator(output="dataset/l1_training_corpus.txt"):
    symbols = string.ascii_letters + string.digits + string.punctuation + " "
    corpus = ""
    for sym in symbols:
        repeat = random.randint(50, 100)  # Repetition for stability
        corpus += sym * repeat + " "  # Space to simulate breaks

    # Shuffle to break sequences
    corpus_list = list(corpus)
    random.shuffle(corpus_list)
    corpus = "".join(corpus_list * 10)  # Multiply for volume (e.g., 100k+ chars)

    with open(output, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"Generated corpus with {len(corpus)} characters.")


if __name__ == "__main__":
    simple_uppercase_char_generator()