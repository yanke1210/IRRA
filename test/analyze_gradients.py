"""Offline gradient analysis on a saved IRRA training run.

Loads configs.yaml + best.pth from a run directory, then measures each
task's gradient on the shared CLIP backbone (base_model) over train
batches. Prints mean L2 norms and pairwise cosine similarities.

Example:
    python test/analyze_gradients.py --output_dir logs/RSTPReid/<run_dir>
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from prettytable import PrettyTable

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets import build_dataloader
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs

LOSS_KEYS = ("sdm_loss", "id_loss", "mlm_loss", "amlm_loss", "itc_loss")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze per-task gradients on the shared CLIP backbone.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Training run directory that contains configs.yaml and the checkpoint.",
    )
    parser.add_argument(
        "--ckpt",
        default="best.pth",
        help="Checkpoint filename inside output_dir (default: best.pth).",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=32,
        help="Number of train batches to average over.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (default: cuda).",
    )
    return parser.parse_args()


def shared_backbone_params(model):
    return [
        p for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("base_model")
    ]


def flatten_grads(params):
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        else:
            chunks.append(p.grad.detach().float().reshape(-1))
    return torch.cat(chunks)


def grad_vec_for_loss(model, params, loss, retain_graph=True):
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain_graph)
    return flatten_grads(params)


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def present_losses(ret):
    keys = []
    for key in LOSS_KEYS:
        value = ret.get(key)
        if torch.is_tensor(value) and value.requires_grad:
            keys.append(key)
    return keys


def analyze(model, train_loader, device, num_batches):
    model.train()
    params = shared_backbone_params(model)
    if not params:
        raise RuntimeError("No trainable base_model parameters found.")

    norm_sums = {}
    cos_sums = {}
    used_keys = None
    n_used = 0

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        batch = move_batch(batch, device)
        ret = model(batch)
        keys = present_losses(ret)
        if not keys:
            raise RuntimeError("Forward pass produced no differentiable * _loss tensors.")
        if used_keys is None:
            used_keys = keys
        elif keys != used_keys:
            raise RuntimeError(
                "Loss set changed across batches: {} vs {}".format(used_keys, keys))

        grads = {}
        for j, key in enumerate(keys):
            retain = j < len(keys) - 1
            grads[key] = grad_vec_for_loss(model, params, ret[key], retain_graph=retain)

        for key, vec in grads.items():
            norm_sums[key] = norm_sums.get(key, 0.0) + vec.norm().item()

        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                pair = (keys[a], keys[b])
                cos = F.cosine_similarity(
                    grads[pair[0]].unsqueeze(0),
                    grads[pair[1]].unsqueeze(0),
                ).item()
                cos_sums[pair] = cos_sums.get(pair, 0.0) + cos

        n_used += 1
        print("batch {}/{} losses: {}".format(
            n_used, num_batches,
            ", ".join("{}={:.4f}".format(k, ret[k].item()) for k in keys)))

    if n_used == 0:
        raise RuntimeError("Train loader yielded no batches.")

    mean_norms = {k: v / n_used for k, v in norm_sums.items()}
    mean_cos = {k: v / n_used for k, v in cos_sums.items()}
    return used_keys, n_used, mean_norms, mean_cos


def print_tables(keys, mean_norms, mean_cos):
    norm_table = PrettyTable(["task", "||g||"])
    norm_table.custom_format["||g||"] = lambda f, v: "{:.4f}".format(v)
    for key in keys:
        norm_table.add_row([key, mean_norms[key]])
    print("\nMean backbone gradient norms")
    print(norm_table)

    if mean_cos:
        cos_table = PrettyTable(["task_a", "task_b", "cosine"])
        cos_table.custom_format["cosine"] = lambda f, v: "{:.4f}".format(v)
        for (a, b), value in mean_cos.items():
            cos_table.add_row([a, b, value])
        print("\nMean pairwise cosine on backbone gradients")
        print(cos_table)


def main():
    cli = parse_args()
    output_dir = os.path.abspath(cli.output_dir)
    config_path = os.path.join(output_dir, "configs.yaml")
    ckpt_path = os.path.join(output_dir, cli.ckpt)
    if not os.path.isfile(config_path):
        raise FileNotFoundError("Missing {}".format(config_path))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError("Missing {}".format(ckpt_path))

    args = load_train_configs(config_path)
    args.training = True
    args.distributed = False
    args.output_dir = output_dir

    device = cli.device
    train_loader, _, _, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    Checkpointer(model).load(f=ckpt_path)
    model.to(device)

    keys, n_used, mean_norms, mean_cos = analyze(
        model, train_loader, device, cli.num_batches)

    print("\nRun: {}".format(output_dir))
    print("Checkpoint: {}".format(ckpt_path))
    print("loss_names: {}".format(args.loss_names))
    print("batches: {}".format(n_used))
    print_tables(keys, mean_norms, mean_cos)

    result = {
        "output_dir": output_dir,
        "ckpt": ckpt_path,
        "loss_names": args.loss_names,
        "num_batches": n_used,
        "grad_norms": mean_norms,
        "cosine": {"{},{}".format(a, b): v for (a, b), v in mean_cos.items()},
    }
    json_path = os.path.join(output_dir, "grad_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\nWrote {}".format(json_path))


if __name__ == "__main__":
    main()
