import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_model import LayerNorm


class OTMatcher(nn.Module):
    """Token matching used by GFGA when concept fusion is off.

    Image and text tokens are projected, then aligned by entropic optimal
    transport. Node masker turns each global vector into a transport marginal.
    Dustbins absorb unmatched mass and are excluded from the score.
    """

    def __init__(self, embed_dim, hid_dim=512, num_dustbins=8,
                 sinkhorn_eps=0.1, sinkhorn_iters=20, gate_init=0.1):
        super().__init__()
        self.hid_dim = int(hid_dim)
        self.num_dustbins = int(num_dustbins)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        self.ot_gate_logit = nn.Parameter(
            torch.tensor(math.log(gate_init / (1.0 - gate_init)), dtype=torch.float32))

        self.img_in_proj = nn.Linear(embed_dim, self.hid_dim)
        self.txt_in_proj = nn.Linear(embed_dim, self.hid_dim)
        self.ln_img = LayerNorm(self.hid_dim)
        self.ln_txt = LayerNorm(self.hid_dim)
        self.img_out_proj = nn.Linear(self.hid_dim, self.hid_dim)
        self.txt_out_proj = nn.Linear(self.hid_dim, self.hid_dim)

        self.img_query = nn.Linear(embed_dim, self.hid_dim)
        self.img_key = nn.Linear(self.hid_dim, self.hid_dim)
        self.txt_query = nn.Linear(embed_dim, self.hid_dim)
        self.txt_key = nn.Linear(self.hid_dim, self.hid_dim)

        self.img_dustbins = nn.Parameter(torch.randn(1, self.num_dustbins, self.hid_dim))
        self.txt_dustbins = nn.Parameter(torch.randn(1, self.num_dustbins, self.hid_dim))
        self.img_dustbin_ln = LayerNorm(self.hid_dim)
        self.txt_dustbin_ln = LayerNorm(self.hid_dim)
        self.img_dustbin_proj = nn.Linear(self.hid_dim, self.hid_dim)
        self.txt_dustbin_proj = nn.Linear(self.hid_dim, self.hid_dim)

    def gate(self):
        return torch.sigmoid(self.ot_gate_logit)

    def sinkhorn(self, mu, mv, cost_matrix):
        mu = mu.float()
        mv = mv.float()
        cost_matrix = cost_matrix.float()
        kernel = torch.exp(-cost_matrix / self.sinkhorn_eps)

        u = torch.ones_like(mu)
        v = torch.ones_like(mv)
        for _ in range(self.sinkhorn_iters):
            u = mu / torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1)
            v = mv / torch.bmm(kernel.permute(0, 2, 1), u.unsqueeze(-1)).squeeze(-1)

        transport = torch.bmm(torch.diag_embed(u), kernel)
        transport = torch.bmm(transport, torch.diag_embed(v))
        return transport

    def forward(self, img_global, img_tokens, txt_global, txt_tokens, lengths):
        batch_size, num_img_nodes, _ = img_tokens.shape
        num_txt_nodes = txt_tokens.shape[1]
        device = img_tokens.device
        dtype = self.img_in_proj.weight.dtype
        lengths = lengths.to(device=device)
        img_global = img_global.to(device=device, dtype=dtype)
        txt_global = txt_global.to(device=device, dtype=dtype)
        img_tokens = img_tokens.to(device=device, dtype=dtype)
        txt_tokens = txt_tokens.to(device=device, dtype=dtype)

        img_tokens = self.img_out_proj(self.ln_img(self.img_in_proj(img_tokens)))
        txt_tokens = self.txt_out_proj(self.ln_txt(self.txt_in_proj(txt_tokens)))

        positions = torch.arange(num_txt_nodes, device=device).unsqueeze(0)
        pad_mask = positions >= lengths.unsqueeze(1)

        img_scores = self._node_scores(self.img_query, self.img_key, img_global, img_tokens, None)
        txt_scores = self._node_scores(self.txt_query, self.txt_key, txt_global, txt_tokens, pad_mask)

        img_dustbins = self.img_dustbins.repeat(batch_size, 1, 1).to(dtype=dtype)
        txt_dustbins = self.txt_dustbins.repeat(batch_size, 1, 1).to(dtype=dtype)
        img_dustbins = self.img_dustbin_proj(self.img_dustbin_ln(img_dustbins))
        txt_dustbins = self.txt_dustbin_proj(self.txt_dustbin_ln(txt_dustbins))
        img_tokens = torch.cat([img_tokens, img_dustbins], dim=1)
        txt_tokens = torch.cat([txt_tokens, txt_dustbins], dim=1)

        dustbin_scores = img_scores.new_ones(batch_size, self.num_dustbins) / self.num_dustbins
        img_scores = torch.cat([img_scores, dustbin_scores], dim=-1)
        txt_scores = torch.cat([txt_scores, dustbin_scores], dim=-1)

        img_tokens = F.normalize(img_tokens.float(), p=2, dim=-1)
        txt_tokens = F.normalize(txt_tokens.float(), p=2, dim=-1)
        sim = torch.matmul(img_tokens, txt_tokens.transpose(1, 2))
        transport = self.sinkhorn(img_scores, txt_scores, 1.0 - sim)
        score = torch.sum(
            transport[:, :num_img_nodes, :num_txt_nodes] * sim[:, :num_img_nodes, :num_txt_nodes],
            dim=(-1, -2),
        )
        return score

    def _node_scores(self, query, key, global_emb, tokens, pad_mask):
        scores = torch.matmul(query(global_emb).unsqueeze(1), key(tokens).transpose(1, 2)).squeeze(1)
        scores = scores.float() / (self.hid_dim ** 0.5)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float('-inf'))
        return torch.softmax(scores, dim=-1)
