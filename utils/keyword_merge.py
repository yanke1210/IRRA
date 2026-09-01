"""Merge and deduplicate per-dataset keyword lists.

Reads the three ``*_keyword_512.txt`` files under ``data/``, unions them
with a set, and writes the unique keywords to ``data/merged_keywords.txt``
(one word per line, sorted).

Example:
    python utils/keyword_merge.py
"""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")

KEYWORD_FILES = (
    "CUHK-PEDES_keyword_512.txt",
    "ICFG-PEDES_keyword_512.txt",
    "RSTPReid_keyword_512.txt",
)
OUTPUT_NAME = "merged_keywords.txt"


def read_keywords(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                words.append(word)
    return words


def main():
    keywords = set()
    for name in KEYWORD_FILES:
        path = os.path.join(_DATA_DIR, name)
        if not os.path.isfile(path):
            raise FileNotFoundError("Keyword file not found: {}".format(path))
        words = read_keywords(path)
        before = len(keywords)
        keywords.update(words)
        print("[{}] {} words, +{} new (set size {})".format(
            name, len(words), len(keywords) - before, len(keywords)))

    out_path = os.path.join(_DATA_DIR, OUTPUT_NAME)
    ranked = sorted(keywords)
    with open(out_path, "w", encoding="utf-8") as f:
        for word in ranked:
            f.write(word + "\n")

    print("saved: {} ({} unique keywords)".format(out_path, len(ranked)))


if __name__ == "__main__":
    main()
