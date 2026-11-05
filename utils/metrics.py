from prettytable import PrettyTable
import torch
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
    return all_cmc


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

    def _compute_ot_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids = [], []
        qfeats, qtokens, qlens = [], [], []
        gfeats, gtokens = [], []
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                global_feat, tokens, lengths = model.encode_text_tokens(caption)
            qids.append(pid.view(-1))
            qfeats.append(global_feat.cpu())
            qtokens.append(tokens.cpu())
            qlens.append(lengths.cpu())
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                global_feat, tokens = model.encode_image_tokens(img)
            gids.append(pid.view(-1))
            gfeats.append(global_feat.cpu())
            gtokens.append(tokens.cpu())

        return (torch.cat(qfeats, 0), torch.cat(qtokens, 0), torch.cat(qlens, 0),
                torch.cat(gfeats, 0), torch.cat(gtokens, 0),
                torch.cat(qids, 0), torch.cat(gids, 0))
    
    def eval(self, model, i2t_metric=False):

        if getattr(model, 'use_ot', False):
            qfeats, qtokens, qlens, gfeats, gtokens, qids, gids = self._compute_ot_embedding(model)
            with torch.no_grad():
                similarity = ot_similarity(
                    model, gfeats, gtokens, qfeats, qtokens, qlens)
        else:
            qfeats, gfeats, qids, gids = self._compute_embedding(model)
            qfeats = F.normalize(qfeats, p=2, dim=1) # text features
            gfeats = F.normalize(gfeats, p=2, dim=1) # image features
            similarity = qfeats @ gfeats.t()

        t2i_cmc = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10)
        i2t_cmc = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10)
        t2i_mr = mean_recall(t2i_cmc)
        i2t_mr = mean_recall(i2t_cmc)
        bidirectional_mr = (t2i_mr + i2t_mr) / 2.0

        table = PrettyTable(["task", "R1", "R5", "R10", "mR"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mr])
        table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mr])
        for name in ("R1", "R5", "R10", "mR"):
            table.custom_format[name] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        self.logger.info("bidirectional mR: {:.3f}".format(bidirectional_mr))

        return bidirectional_mr


def ot_similarity(model, img_global, img_tokens, txt_global, txt_tokens, txt_lengths):
    """Rescore the union of both global top-k lists.

    Candidate pairs become global cosine plus the transport score. Every
    other pair is -100, matching GFGA's shortlist retrieval.
    """
    device = next(model.parameters()).device
    img_global = F.normalize(img_global.float(), p=2, dim=-1).to(device)
    txt_global = F.normalize(txt_global.float(), p=2, dim=-1).to(device)
    sim = img_global @ txt_global.t()

    k = min(int(model.ot_k_candidates), sim.shape[0], sim.shape[1])
    if k <= 0:
        return sim.t()

    _, i2t_topk = torch.topk(sim, k=k, dim=-1)
    _, t2i_topk = torch.topk(sim, k=k, dim=0)
    t2i_topk = t2i_topk.t()

    img_ids = torch.arange(sim.shape[0], device=device).unsqueeze(1).expand_as(i2t_topk)
    i2t_pairs = torch.stack([img_ids.reshape(-1), i2t_topk.reshape(-1)], dim=1)
    txt_ids = torch.arange(sim.shape[1], device=device).unsqueeze(1).expand_as(t2i_topk)
    t2i_pairs = torch.stack([t2i_topk.reshape(-1), txt_ids.reshape(-1)], dim=1)
    candidates = torch.unique(torch.cat([i2t_pairs, t2i_pairs], dim=0), dim=0)

    local_sim = []
    batch_size = max(int(model.ot_eval_batch_size), 1)
    for begin in range(0, candidates.shape[0], batch_size):
        cand = candidates[begin:begin + batch_size]
        cand_cpu = cand.cpu()
        local_sim.append(model.ot_matcher(
            img_global[cand[:, 0]],
            img_tokens[cand_cpu[:, 0]].to(device),
            txt_global[cand[:, 1]],
            txt_tokens[cand_cpu[:, 1]].to(device),
            txt_lengths[cand_cpu[:, 1]].to(device),
        ))
    local_sim = torch.cat(local_sim, dim=0)

    final = torch.full_like(sim, -100.0)
    gate = model.ot_matcher.gate().to(dtype=sim.dtype)
    chosen = sim[candidates[:, 0], candidates[:, 1]]
    final[candidates[:, 0], candidates[:, 1]] = chosen + gate * local_sim
    return final.t()
