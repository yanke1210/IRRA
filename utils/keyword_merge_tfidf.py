"""Merge TF-IDF per-dataset keyword lists.

Reads data/*_keyword_tfidf.txt and writes data/merged_keywords_tfidf.txt.
Does not touch data/merged_keywords.txt.

Example:
    python utils/keyword_merge_tfidf.py
"""
import argparse
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")

KEYWORD_FILES = (
    "CUHK-PEDES_keyword_tfidf.txt",
    "ICFG-PEDES_keyword_tfidf.txt",
    "RSTPReid_keyword_tfidf.txt",
)
OUTPUT_NAME = "merged_keywords_tfidf.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-dataset TF-IDF keyword files.")
    parser.add_argument(
        "--data_dir",
        default=_DATA_DIR,
        help="Directory containing *_keyword_tfidf.txt files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite merged_keywords_tfidf.txt if it exists.",
    )
    return parser.parse_args()


def read_keywords(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                words.append(word)
    return words


def main():
    args = parse_args()
    out_path = os.path.join(args.data_dir, OUTPUT_NAME)
    if os.path.isfile(out_path) and not args.force:
        raise FileExistsError(
            "Refusing to overwrite {} (pass --force to replace).".format(out_path))

    keywords = set()
    for name in KEYWORD_FILES:
        path = os.path.join(args.data_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError("Keyword file not found: {}".format(path))
        words = read_keywords(path)
        before = len(keywords)
        keywords.update(words)
        print("[{}] {} words, +{} new (set size {})".format(
            name, len(words), len(keywords) - before, len(keywords)))

    ranked = sorted(keywords)
    os.makedirs(args.data_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for word in ranked:
            f.write(word + "\n")

    print("saved: {} ({} unique keywords)".format(out_path, len(ranked)))


if __name__ == "__main__":
    main()
