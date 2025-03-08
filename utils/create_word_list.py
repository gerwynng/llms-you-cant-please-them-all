import argparse
import os
import string
from collections import Counter

import pandas as pd


def compute_doc_freq(corpus: list[str]) -> dict[str, int]:
    doc_freq: dict[str, int] = Counter()
    for text in corpus:
        # Tokenize & clean the words in this document
        words = [
            w.strip(string.punctuation).lower()
            for w in text.split()
            if w.strip(string.punctuation)
        ]
        # Make a unique set of words for this document
        unique_words_in_doc = set(words)
        for word in unique_words_in_doc:
            doc_freq[word] += 1
    return doc_freq


def filter_by_df(
    doc_freq: dict[str, int],
    max_df: int,
    max_word_length: int,
    min_word_length: int,
) -> list[str]:
    filtered: list[str] = []
    for k, v in doc_freq.items():
        word_length = len(k)
        if word_length < min_word_length or word_length > max_word_length:
            continue
        if v > max_df:
            continue
        if all(c.isdigit() for c in k):
            continue
        if all(c.isalpha() for c in k):
            filtered.append(k)
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter words by document frequency.",
    )
    parser.add_argument(
        "--brown_file",
        type=str,
        default="./data/brown.csv",
        help="Path to the brown file.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="./data/filtered_brown_max_df_isalpha_only_v2.txt",
        help="Path for the output file.",
    )
    parser.add_argument(
        "--max_df", type=int, default=3, help="Maximum document frequency."
    )
    parser.add_argument(
        "--min_word_length", type=int, default=0, help="Minimum word length."
    )
    parser.add_argument(
        "--max_word_length", type=int, default=100, help="Maximum word length."
    )
    args = parser.parse_args()

    # Read the data
    brown = pd.read_csv(args.brown_file)
    corpus: list[str] = brown["tokenized_text"].tolist()

    # Compute document frequency
    doc_freq = compute_doc_freq(corpus=corpus)

    # Filter by document frequency and other constraints
    filtered_word_list = filter_by_df(
        doc_freq=doc_freq,
        max_df=args.max_df,
        min_word_length=args.min_word_length,
        max_word_length=args.max_word_length,
    )

    print(f"filtered word list: {len(filtered_word_list)} words")

    # Write out results
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        for word in sorted(filtered_word_list):
            f.write(word + "\n")


if __name__ == "__main__":
    main()
