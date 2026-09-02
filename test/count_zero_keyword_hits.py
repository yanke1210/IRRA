"""Count train captions with zero keyword hits.

Uses the same train-split JSON loading as utils/keyword_statistic.py and the
same extract_keywords() used at training time.

Example:
    python test/count_zero_keyword_hits.py --root_dir D:/datasets
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.bases import ImageTextMLMDataset
from datasets.bases import tokenize as clip_tokenize
from utils.keyword_statistic import (
    DATASETS,
    iter_train_captions,
    load_annos,
    resolve_anno_path,
)
from utils.simple_tokenizer import SimpleTokenizer, extract_keywords


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count train captions with zero keyword hits.")
    parser.add_argument(
        "--root_dir",
        default="./data",
        help="Dataset root that contains CUHK-PEDES / ICFG-PEDES / RSTPReid.",
    )
    return parser.parse_args()


def count_dataset(name, json_names, root_dir, mapper):
    dataset_dir = os.path.join(root_dir, name)
    anno_path = resolve_anno_path(dataset_dir, json_names)
    captions = list(iter_train_captions(load_annos(anno_path)))

    zero_hits = 0
    mapped_empty = 0
    for caption in captions:
        keyword_positions = extract_keywords(caption)
        if not keyword_positions:
            zero_hits += 1
            continue
        caption_tokens = clip_tokenize(
            caption, tokenizer=mapper.tokenizer, text_length=77, truncate=True)
        token_positions = mapper._map_keyword_positions_to_tokens(
            keyword_positions, caption_tokens)
        if not token_positions:
            mapped_empty += 1

    total = len(captions)
    hit_captions = total - zero_hits
    print("[{}]".format(name))
    print("  json           : {}".format(anno_path))
    print("  split          : train")
    print("  captions       : {}".format(total))
    print("  zero keyword   : {} ({:.2f}%)".format(
        zero_hits, 100.0 * zero_hits / total if total else 0.0))
    print("  hit but no mask: {} ({:.2f}% of hit captions)".format(
        mapped_empty,
        100.0 * mapped_empty / hit_captions if hit_captions else 0.0))
    print("")
    return total, zero_hits, mapped_empty


def main():
    args = parse_args()
    mapper = ImageTextMLMDataset.__new__(ImageTextMLMDataset)
    mapper.tokenizer = SimpleTokenizer()

    totals = [0, 0, 0]
    for name, json_names in DATASETS.items():
        total, zero_hits, mapped_empty = count_dataset(
            name, json_names, args.root_dir, mapper)
        totals[0] += total
        totals[1] += zero_hits
        totals[2] += mapped_empty

    total, zero_hits, mapped_empty = totals
    hit_captions = total - zero_hits
    print("[all]")
    print("  captions       : {}".format(total))
    print("  zero keyword   : {} ({:.2f}%)".format(
        zero_hits, 100.0 * zero_hits / total if total else 0.0))
    print("  hit but no mask: {} ({:.2f}% of hit captions)".format(
        mapped_empty,
        100.0 * mapped_empty / hit_captions if hit_captions else 0.0))


if __name__ == "__main__":
    main()
