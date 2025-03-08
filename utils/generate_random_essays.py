import argparse
import json
import random
import string

from lingua import Language, LanguageDetector, LanguageDetectorBuilder
from tqdm import tqdm

random.seed(42)


def compute_english_confidence(text: str, detector: LanguageDetector):
    return detector.compute_language_confidence(text, Language.ENGLISH)


def load_word_list(filepath: str) -> list[str]:
    """
    Load word list, keep unqiue and sort for determinism
    """
    word_list: list[str] = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            words = [w.strip(string.punctuation).lower() for w in line.split()]
            word_list += words
    return sorted(list(set(word_list)))


def generate_essays(
    word_list: list[str],
    num_words: int,
    max_char: int = 450,
) -> str:
    essay = " ".join(random.choices(word_list, k=num_words))
    essay = essay[:max_char].strip()
    return essay


def main():
    parser = argparse.ArgumentParser(
        description="Generate random essays from input word list txt.",
    )
    parser.add_argument(
        "--word_list_txt",
        type=str,
        help="Input word list in .txt.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="./data/random_essays.json",
        help="Output list of generated essays in .json.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of essays to generate. (default: 10000)",
    )
    parser.add_argument(
        "--n_words",
        type=int,
        default=50,
        help="Number of words per essay. (default: 50)",
    )
    parser.add_argument(
        "--max_char",
        type=int,
        default=450,
        help="Max character in essays to truncate to. (default: 450)",
    )
    args = parser.parse_args()

    # Init detector
    detector = LanguageDetectorBuilder.from_all_languages().build()

    word_list = load_word_list(args.word_list_txt)

    essays: list[str] = []

    for _ in tqdm(range(args.n_samples), "essay generation loop"):
        essay = generate_essays(
            word_list=word_list,
            num_words=args.n_words,
            max_char=450,
        )
        en_conf_score = compute_english_confidence(essay, detector)
        if en_conf_score == 1.0:
            essays.append(essay)

    print(f"Generated {len(essays)} essays after filtering")

    print(f"sample essay: {essays[0]}")

    with open(args.out_file, "w") as f:
        json.dump(essays, f)

    print(f"Successfully saved essays to {args.out_file}")


if __name__ == "__main__":
    main()
