"""Count train captions with zero keyword hits.

Reads RSICD and RSITMD annotations and uses the same BPE-span match as AMLM.

Example:
    python test/count_zero_keyword_hits.py --root_dir D:/datasets
"""
import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.simple_tokenizer import SimpleTokenizer, keyword_bpe_spans

DATASETS = {
    "RSICD": ("dataset_rsicd.json", "RSICD_keyword_512_refined.txt"),
    "RSITMD": ("dataset_RSITMD.json", "RSITMD_keyword_512_refined.txt"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count train captions with zero keyword hits.")
    parser.add_argument(
        "--root_dir",
        default="./data",
        help="Dataset root that contains RSICD / RSITMD.",
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
    records = annos.get("images", []) if isinstance(annos, dict) else annos
    for anno in records:
        if anno.get("split") != "train":
            continue
        sentences = anno.get("sentences")
        if sentences:
            for sent in sentences:
                text = sent.get("raw") if isinstance(sent, dict) else sent
                if text:
                    yield text
            continue
        captions = anno.get("captions", [])
        if isinstance(captions, str):
            captions = [captions]
        for caption in captions:
            if caption:
                yield caption


def count_dataset(name, json_names, keyword_name, root_dir, tokenizer):
    dataset_dir = os.path.join(root_dir, name)
    anno_path = resolve_anno_path(dataset_dir, json_names)
    keyword_file = os.path.join(_REPO_ROOT, "data", keyword_name)
    captions = list(iter_train_captions(load_annos(anno_path)))

    zero_hits = 0
    for caption in captions:
        spans = keyword_bpe_spans(
            caption, tokenizer, keyword_file=keyword_file, text_length=77)
        if not spans:
            zero_hits += 1

    total = len(captions)
    print("[{}]".format(name))
    print("  json         : {}".format(anno_path))
    print("  keywords     : {}".format(keyword_file))
    print("  split        : train")
    print("  captions     : {}".format(total))
    print("  zero keyword : {} ({:.2f}%)".format(
        zero_hits, 100.0 * zero_hits / total if total else 0.0))
    print("")
    return total, zero_hits


def main():
    args = parse_args()
    tokenizer = SimpleTokenizer()

    totals = [0, 0]
    for name, (json_name, keyword_name) in DATASETS.items():
        total, zero_hits = count_dataset(
            name, (json_name,), keyword_name, args.root_dir, tokenizer)
        totals[0] += total
        totals[1] += zero_hits

    total, zero_hits = totals
    print("[all]")
    print("  captions     : {}".format(total))
    print("  zero keyword : {} ({:.2f}%)".format(
        zero_hits, 100.0 * zero_hits / total if total else 0.0))


if __name__ == "__main__":
    main()
