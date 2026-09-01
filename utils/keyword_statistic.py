"""Count frequent keywords from text-to-image person ReID captions.

Reads the annotation JSON of CUHK-PEDES, ICFG-PEDES and RSTPReid, collects
train-split captions, tokenizes them, drops function words (articles,
prepositions, conjunctions, pronouns, auxiliary verbs), and writes the top-K
keywords of each dataset to ``data/<dataset_name>_keyword_<K>.txt``
(one word per line).

Example:
    python utils/keyword_statistic.py --root_dir D:/datasets
"""
import argparse
import json
import os
import re
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "data")


# Conservative English function-word list (NLTK-style). Content words that
# matter for person descriptions (colors, garments, "top", "front", "long")
# are intentionally kept.
STOPWORDS = {
    "a", "an", "the",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into", "onto",
    "upon", "over", "under", "above", "below", "between", "among", "through",
    "across", "around", "about", "against", "along", "behind", "beside",
    "besides", "beyond", "during", "except", "inside", "outside", "near",
    "off", "toward", "towards", "via", "within", "without", "up", "down",
    "out", "before", "after",
    "and", "or", "but", "nor", "so", "yet", "both", "either", "neither",
    "whether", "than", "as", "if", "because", "until", "while",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "done",
    "has", "have", "had", "having",
    "can", "could", "should", "would", "may", "might", "must", "will", "shall",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "each", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "own", "same", "too", "very", "just", "also",
    "again", "further", "then", "once", "now",
    "s", "t", "d", "ll", "ve", "re", "m", "don", "nt",
}

TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")

DATASETS = {
    "CUHK-PEDES": ("reid_raw.json",),
    "ICFG-PEDES": ("ICFG-PEDES.json", "ICFG_PEDES.json"),
    "RSTPReid": ("data_captions.json",),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistic top-K keywords from person ReID captions.")
    parser.add_argument(
        "--root_dir",
        default="./data",
        help="Dataset root that contains CUHK-PEDES / ICFG-PEDES / RSTPReid.",
    )
    parser.add_argument(
        "--output_dir",
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory to write <dataset>_keyword_<K>.txt files. Default: ./data",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=512,
        help="Number of most frequent keywords to keep.",
    )
    return parser.parse_args()


def resolve_anno_path(dataset_dir, candidates):
    for name in candidates:
        path = os.path.join(dataset_dir, name)
        if os.path.isfile(path):
            return path
    expected = ", ".join(candidates)
    raise FileNotFoundError(
        "Cannot find annotation JSON under '{}'. Tried: {}".format(
            dataset_dir, expected))


def load_annos(anno_path):
    with open(anno_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_train_captions(annos):
    for anno in annos:
        if anno.get("split") != "train":
            continue
        captions = anno.get("captions", [])
        if isinstance(captions, str):
            captions = [captions]
        for caption in captions:
            if caption:
                yield caption


def tokenize(caption):
    # RSTPReid captions often miss spaces after punctuation, e.g. "shirt.His".
    return TOKEN_RE.findall(caption.lower())


def count_keywords(captions):
    counter = Counter()
    num_captions = 0
    for caption in captions:
        num_captions += 1
        for token in tokenize(caption):
            if token in STOPWORDS or len(token) < 2:
                continue
            counter[token] += 1
    return counter, num_captions


def write_keyword_file(path, keywords):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for word in keywords:
            f.write(word + "\n")


def statistic_dataset(name, json_names, root_dir, output_dir, top_k):
    dataset_dir = os.path.join(root_dir, name)
    anno_path = resolve_anno_path(dataset_dir, json_names)
    annos = load_annos(anno_path)
    counter, num_captions = count_keywords(iter_train_captions(annos))

    ranked = counter.most_common(top_k)
    keywords = [word for word, _ in ranked]
    out_name = "{}_keyword_{}.txt".format(name, top_k)
    out_path = os.path.join(output_dir, out_name)
    write_keyword_file(out_path, keywords)

    print("[{}]".format(name))
    print("  json      : {}".format(anno_path))
    print("  split     : train")
    print("  captions  : {}".format(num_captions))
    print("  unique    : {} (after stopword filter)".format(len(counter)))
    print("  saved     : {} (top {})".format(out_path, len(keywords)))
    preview = ", ".join("{}:{}".format(w, c) for w, c in ranked[:15])
    print("  top-15    : {}".format(preview))
    print("")
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for name, json_names in DATASETS.items():
        statistic_dataset(
            name=name,
            json_names=json_names,
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
