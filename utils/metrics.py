from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10):
    indices = torch.argsort(similarity, dim=1, descending=True)
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    tmp_cmc = matches.cumsum(1)  # q * k
    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    return all_cmc, mINP, indices


def mean_recall(cmc):
    """mR = (R@1 + R@5 + R@10) / 3."""
    return (float(cmc[0]) + float(cmc[4]) + float(cmc[9])) / 3.0


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False, num_bins=0):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10)
        i2t_cmc, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10)
        t2i_mr = mean_recall(t2i_cmc)
        i2t_mr = mean_recall(i2t_cmc)
        bidirectional_mr = (t2i_mr + i2t_mr) / 2.0

        table = PrettyTable(["task", "R1", "R5", "R10", "mR", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mr, t2i_mINP])
        table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mr, i2t_mINP])
        table.add_row([
            'mean',
            (float(t2i_cmc[0]) + float(i2t_cmc[0])) / 2.0,
            (float(t2i_cmc[4]) + float(i2t_cmc[4])) / 2.0,
            (float(t2i_cmc[9]) + float(i2t_cmc[9])) / 2.0,
            bidirectional_mr,
            (float(t2i_mINP) + float(i2t_mINP)) / 2.0,
        ])
        for name in ("R1", "R5", "R10", "mR", "mINP"):
            table.custom_format[name] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))

        if num_bins:
            self.last_i2t_gap = mean_calibration_gap(
                similarity.t(), gids, qids, num_bins)
            self.last_t2i_gap = mean_calibration_gap(
                similarity, qids, gids, num_bins)

        return bidirectional_mr


def mean_calibration_gap(scores, query_pids, gallery_pids, num_bins=15):
    """Equal-frequency mean calibration gap used to update TC gamma.

    ``scores`` is a query-gallery similarity matrix. A prediction is correct
    when the top gallery id matches the query id, which covers multiple
    captions of the same image.
    """
    if scores.numel() == 0 or num_bins <= 0:
        return 0.0
    probs = torch.softmax(scores.float(), dim=1)
    confidence, pred = probs.max(dim=1)
    correct = gallery_pids[pred].eq(query_pids).float().detach().cpu()
    confidence = confidence.detach().cpu()
    order = np.sort(confidence.numpy())
    edges = np.interp(
        np.linspace(0, len(order), num_bins + 1),
        np.arange(len(order)),
        order,
    )
    gaps = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = confidence.gt(float(lower)) & confidence.le(float(upper))
        if float(in_bin.float().sum()) > 0:
            accuracy = correct[in_bin].mean()
            average_confidence = confidence[in_bin].mean()
            gaps.append(float(average_confidence - accuracy))
        else:
            gaps.append(0.0)
    if not gaps:
        return 0.0
    return sum(gaps) / len(gaps)
