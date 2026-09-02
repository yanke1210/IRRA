"""Count captions whose CLIP token sequence exceeds length 77.

Length is measured the same way as training tokenize():
    [<|startoftext|>] + encode(caption) + [<|endoftext|>]
before truncation.

Example:
    python test/count_long_captions.py --root_dir D:/datasets
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.keyword_statistic import DATASETS, load_annos, resolve_anno_path
from utils.simple_tokenizer import SimpleTokenizer

TEXT_LENGTH = 77


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count captions longer than CLIP context length 77.")
    parser.add_argument(
        "--root_dir",
        default="./data",
        help="Dataset root that contains CUHK-PEDES / ICFG-PEDES / RSTPReid.",
    )
    parser.add_argument(
        "--splits",
        default="all",
        choices=["all", "train", "test", "val"],
        help="Which split(s) to count. Default: all captions.",
    )
    return parser.parse_args()


def iter_captions(annos, splits):
    for anno in annos:
        if splits != "all" and anno.get("split") != splits:
            continue
        captions = anno.get("captions", [])
        if isinstance(captions, str):
            captions = [captions]
        for caption in captions:
            if caption:
                yield caption, anno.get("split", "")


def clip_token_length(caption, tokenizer):
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    return 2 + len(tokenizer.encode(caption)), sot, eot


def count_dataset(name, json_names, root_dir, splits, tokenizer):
    dataset_dir = os.path.join(root_dir, name)
    anno_path = resolve_anno_path(dataset_dir, json_names)
    annos = load_annos(anno_path)

    total = 0
    over = 0
    max_len = 0
    examples = []
    for caption, split in iter_captions(annos, splits):
        total += 1
        length, _, _ = clip_token_length(caption, tokenizer)
        if length > max_len:
            max_len = length
        if length > TEXT_LENGTH:
            over += 1
            if len(examples) < 3:
                examples.append((length, split, caption[:120]))

    print("[{}]".format(name))
    print("  json        : {}".format(anno_path))
    print("  split       : {}".format(splits))
    print("  captions    : {}".format(total))
    print("  over 77     : {} ({:.2f}%)".format(
        over, 100.0 * over / total if total else 0.0))
    print("  max length  : {}".format(max_len))
    for length, split, text in examples:
        print("  example     : len={} split={} {}".format(length, split, text))
    print("")
    return total, over, max_len


def main():
    args = parse_args()
    tokenizer = SimpleTokenizer()

    totals = [0, 0, 0]
    for name, json_names in DATASETS.items():
        total, over, max_len = count_dataset(
            name, json_names, args.root_dir, args.splits, tokenizer)
        totals[0] += total
        totals[1] += over
        totals[2] = max(totals[2], max_len)

    total, over, max_len = totals
    print("[all]")
    print("  captions    : {}".format(total))
    print("  over 77     : {} ({:.2f}%)".format(
        over, 100.0 * over / total if total else 0.0))
    print("  max length  : {}".format(max_len))


if __name__ == "__main__":
    main()
