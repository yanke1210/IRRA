"""Select distinctive person-ReID keywords with TF-IDF.

Reads train-split captions of CUHK-PEDES / ICFG-PEDES / RSTPReid, drops
function words and generic description verbs/nouns, drops terms whose
document frequency exceeds a ratio of captions, ranks the rest by TF-IDF,
and writes ``data/<dataset>_keyword_tfidf.txt``.

Does not overwrite existing files unless --force is set.
Does not write data/merged_keywords.txt.

Example:
    python utils/keyword_statistic_tfidf.py --root_dir D:/datasets
"""
import argparse
import math
import os
from collections import Counter

from keyword_statistic import (
    DATASETS,
    STOPWORDS,
    TOKEN_RE,
    _DEFAULT_OUTPUT_DIR,
    iter_train_captions,
    load_annos,
    resolve_anno_path,
)

# Words that dominate person captions but barely distinguish identities.
GENERIC_WORDS = {
    "wearing", "wears", "wear", "worn",
    "man", "woman", "person", "people", "guy", "lady", "girl", "boy",
    "male", "female", "pedestrian", "someone", "somebody",
    "walking", "walks", "walk", "standing", "stands", "stand",
    "going", "looks", "looking", "appears", "appear",
    "pair", "pairs", "carrying", "carries", "carry",
    "dressed", "clad", "attired",
    "there", "also", "one", "two", "another",
}

MAX_DF_RATIO = 0.30
DEFAULT_TOP_K = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description="TF-IDF keywords from person ReID train captions.")
    parser.add_argument(
        "--root_dir",
        default="./data",
        help="Dataset root that contains CUHK-PEDES / ICFG-PEDES / RSTPReid.",
    )
    parser.add_argument(
        "--output_dir",
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for <dataset>_keyword_tfidf.txt. Default: ./data",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of TF-IDF keywords to keep per dataset.",
    )
    parser.add_argument(
        "--max_df_ratio",
        type=float,
        default=MAX_DF_RATIO,
        help="Drop terms that appear in more than this fraction of captions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def tokenize(caption):
    return TOKEN_RE.findall(caption.lower())


def is_content_word(token):
    if len(token) < 2:
        return False
    if token in STOPWORDS or token in GENERIC_WORDS:
        return False
    return True


def caption_terms(caption):
    return [t for t in tokenize(caption) if is_content_word(t)]


def tfidf_keywords(captions, top_k, max_df_ratio):
    docs = []
    tf = Counter()
    df = Counter()
    for caption in captions:
        terms = caption_terms(caption)
        if not terms:
            continue
        docs.append(terms)
        tf.update(terms)
        df.update(set(terms))

    n_docs = len(docs)
    if n_docs == 0:
        return [], n_docs, 0

    max_df = max_df_ratio * n_docs
    scored = []
    for word, term_freq in tf.items():
        doc_freq = df[word]
        if doc_freq > max_df:
            continue
        idf = math.log(float(n_docs) / doc_freq)
        scored.append((word, term_freq * idf, term_freq, doc_freq))
    scored.sort(key=lambda x: (-x[1], x[0]))
    ranked = scored[:top_k]
    return ranked, n_docs, len(tf)


def write_keywords(path, words, force):
    if os.path.isfile(path) and not force:
        raise FileExistsError(
            "Refusing to overwrite {} (pass --force to replace).".format(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")


def statistic_dataset(name, json_names, args):
    dataset_dir = os.path.join(args.root_dir, name)
    anno_path = resolve_anno_path(dataset_dir, json_names)
    captions = list(iter_train_captions(load_annos(anno_path)))
    ranked, n_docs, n_unique = tfidf_keywords(
        captions, args.top_k, args.max_df_ratio)
    keywords = [word for word, _, _, _ in ranked]
    out_path = os.path.join(args.output_dir, "{}_keyword_tfidf.txt".format(name))
    write_keywords(out_path, keywords, args.force)

    print("[{}]".format(name))
    print("  json      : {}".format(anno_path))
    print("  captions  : {}".format(n_docs))
    print("  unique    : {} (after filters)".format(n_unique))
    print("  saved     : {} (top {})".format(out_path, len(keywords)))
    preview = ", ".join("{}:{:.1f}".format(w, s) for w, s, _, _ in ranked[:15])
    print("  top-15    : {}".format(preview))
    print("")
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for name, json_names in DATASETS.items():
        statistic_dataset(name, json_names, args)


if __name__ == "__main__":
    main()
