from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in self.current_task:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if self.use_mlm or self.use_amlm:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        if 'ker' in self.current_task:
            self.current_task.extend(['amlm', 'ot'])
        if 'irr' in self.current_task:
            self.current_task.extend(['sdm', 'mlm'])
        self.current_task = list(dict.fromkeys(self.current_task))
        self.use_mlm = 'mlm' in self.current_task
        self.use_amlm = 'amlm' in self.current_task
        self.use_ot = 'ot' in self.current_task
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if self.use_mlm:
            mlm_scores = self._mlm_scores(batch['mlm_ids'], image_feats)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({
                'mlm_loss': objectives.compute_mlm(mlm_scores, mlm_labels)
                            * self.args.mlm_loss_weight
            })
            ret.update({'mlm_acc': self._masked_token_acc(mlm_scores, mlm_labels)})

        if self.use_amlm:
            amlm_text_feats = self.base_model.encode_text(batch['amlm_ids'])
            amlm_hidden = self.cross_former(amlm_text_feats, image_feats, image_feats)
            amlm_scores = self.mlm_head(amlm_hidden).float().reshape(-1, self.args.vocab_size)
            amlm_labels = batch['amlm_labels'].reshape(-1)
            ret.update({
                'amlm_loss': objectives.compute_mlm(amlm_scores, amlm_labels)
                             * self.args.amlm_loss_weight
            })
            ret.update({'amlm_acc': self._masked_token_acc(amlm_scores, amlm_labels)})

            if self.use_ot:
                ot_text_feats = self.base_model.encode_text(batch['ot_ids'])
                keyword_feats, keyword_mask = self._pool_keyword_spans(
                    ot_text_feats, batch['keyword_groups'])
                patch_feats = image_feats[:, 1:, :]
                ret.update({
                    'ot_loss': objectives.compute_ot(
                        keyword_feats, patch_feats, keyword_mask=keyword_mask,
                        pids=batch['pids'],
                        epsilon=getattr(self.args, 'ot_epsilon', 0.1),
                        sinkhorn_iterations=getattr(
                            self.args, 'ot_sinkhorn_iterations', 20),
                        partial_ratio=getattr(
                            self.args, 'ot_partial_ratio', 0.25),
                        margin=getattr(self.args, 'ot_margin', 0.2),
                    ) * getattr(self.args, 'ot_loss_weight', 1.0)
                })

        return ret

    def _mlm_scores(self, mlm_ids, image_feats):
        x = self._mlm_features(mlm_ids, image_feats)
        x = self.mlm_head(x)
        return x.float().reshape(-1, self.args.vocab_size)

    def _mlm_features(self, mlm_ids, image_feats):
        mlm_feats = self.base_model.encode_text(mlm_ids)
        x = self.cross_former(mlm_feats, image_feats, image_feats)
        return x

    @staticmethod
    def _pool_keyword_spans(text_feats, keyword_groups):
        """Pool all BPE pieces belonging to the same keyword word."""
        batch_keywords = []
        lengths = []
        for features, groups in zip(text_feats, keyword_groups):
            pooled = [
                features[groups == group_id].mean(dim=0)
                for group_id in range(1, int(groups.max().item()) + 1)
            ]
            lengths.append(len(pooled))
            if pooled:
                batch_keywords.append(torch.stack(pooled))
            else:
                batch_keywords.append(features.new_zeros((0, features.shape[-1])))

        max_keywords = max(max(lengths), 1)
        output = text_feats.new_zeros(
            (text_feats.shape[0], max_keywords, text_feats.shape[-1]))
        mask = torch.zeros(
            (text_feats.shape[0], max_keywords), dtype=torch.bool,
            device=text_feats.device)
        for index, pooled in enumerate(batch_keywords):
            if pooled.shape[0]:
                output[index, :pooled.shape[0]] = pooled
                mask[index, :pooled.shape[0]] = True
        return output, mask

    def _masked_token_acc(self, scores, labels):
        pred = scores.max(1)[1]
        label_idx = torch.nonzero(labels)
        if label_idx.numel() == 0:
            return scores.new_zeros(())
        return (pred[label_idx] == labels[label_idx]).float().mean()


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
