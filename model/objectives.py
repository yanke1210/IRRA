import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    # ignore_index=0 drops unmasked positions; if a batch has no valid
    # targets (possible for keyword masking), CE would be nan.
    if (labels != 0).sum() == 0:
        return scores.new_zeros(())
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def _partial_ot_distance(keyword_features, patch_features, partial_ratio,
                         epsilon, sinkhorn_iterations):
    """Sinkhorn distance after retaining the most relevant patch subset."""
    keyword_features = F.normalize(keyword_features.float(), dim=-1)
    patch_features = F.normalize(patch_features.float(), dim=-1)
    cost = (1.0 - keyword_features @ patch_features.t()).clamp_min(0.0)

    num_patches = cost.shape[1]
    num_selected = max(1, min(num_patches,
                              int(math.ceil(num_patches * partial_ratio))))
    patch_relevance = cost.min(dim=0).values
    selected = patch_relevance.topk(num_selected, largest=False).indices
    cost = cost[:, selected]

    keyword_mass = cost.new_full((cost.shape[0],), 1.0 / cost.shape[0])
    patch_mass = cost.new_full((cost.shape[1],), 1.0 / cost.shape[1])
    kernel = torch.exp(-cost / max(epsilon, 1e-6)).clamp_min(1e-12)
    transport = kernel / kernel.sum().clamp_min(1e-12)
    for _ in range(sinkhorn_iterations):
        transport = transport * (
            keyword_mass / transport.sum(dim=1).clamp_min(1e-12)).unsqueeze(1)
        transport = transport * (
            patch_mass / transport.sum(dim=0).clamp_min(1e-12)).unsqueeze(0)
    return (transport * cost).sum()


def compute_ot(keyword_features, patch_features, keyword_mask=None,
               patch_mask=None, pids=None, epsilon=0.1,
               sinkhorn_iterations=20, partial_ratio=0.25, margin=0.2):
    """Bidirectional hard-negative partial OT ranking loss.

    Keyword features must be text-only representations (before cross-modal
    attention).  Hard negatives are selected in-batch after excluding samples
    with the same person identity.
    """
    if keyword_features.ndim != 3 or patch_features.ndim != 3:
        raise ValueError("OT features must have shape [batch, tokens, dimension]")
    if not 0.0 < partial_ratio <= 1.0:
        raise ValueError("partial_ratio must be in (0, 1]")

    batch_size = keyword_features.shape[0]
    if keyword_mask is None:
        keyword_mask = torch.ones(keyword_features.shape[:2], dtype=torch.bool,
                                  device=keyword_features.device)
    if patch_mask is None:
        patch_mask = torch.ones(patch_features.shape[:2], dtype=torch.bool,
                                device=patch_features.device)

    valid_keywords = []
    valid_patches = []
    for batch_index in range(batch_size):
        keyword_valid = keyword_features[batch_index][keyword_mask[batch_index]]
        patch_valid = patch_features[batch_index][patch_mask[batch_index]]
        valid_keywords.append(keyword_valid)
        valid_patches.append(patch_valid)

    usable = [
        index for index in range(batch_size)
        if valid_keywords[index].shape[0] and valid_patches[index].shape[0]
    ]
    if len(usable) < 2:
        return keyword_features.sum() * 0.0

    if pids is None:
        pids = torch.arange(batch_size, device=keyword_features.device)
    else:
        pids = pids.reshape(-1)

    # Coarse text/patch similarity only chooses the hard-negative indices.  OT
    # itself remains the optimized fine-grained distance.
    text_pooled = torch.stack([
        F.normalize(valid_keywords[i].float(), dim=-1).mean(dim=0)
        if valid_keywords[i].shape[0] else keyword_features.new_zeros(keyword_features.shape[-1]).float()
        for i in range(batch_size)
    ])
    image_pooled = torch.stack([
        F.normalize(valid_patches[i].float(), dim=-1).mean(dim=0)
        if valid_patches[i].shape[0] else patch_features.new_zeros(patch_features.shape[-1]).float()
        for i in range(batch_size)
    ])
    coarse_similarity = F.normalize(text_pooled, dim=-1) @ F.normalize(image_pooled, dim=-1).t()
    invalid_negative = pids[:, None].eq(pids[None, :])
    unusable = torch.ones(batch_size, dtype=torch.bool,
                          device=keyword_features.device)
    unusable[usable] = False
    invalid_negative = invalid_negative | unusable[:, None] | unusable[None, :]
    coarse_similarity = coarse_similarity.masked_fill(invalid_negative, float("-inf"))

    losses = []
    for index in usable:
        if torch.isneginf(coarse_similarity[index]).all():
            continue
        negative_image = coarse_similarity[index].argmax().item()
        negative_text = coarse_similarity[:, index].argmax().item()
        if (valid_patches[negative_image].shape[0] == 0 or
                valid_keywords[negative_text].shape[0] == 0):
            continue

        positive = _partial_ot_distance(
            valid_keywords[index], valid_patches[index], partial_ratio,
            epsilon, sinkhorn_iterations)
        text_to_image_negative = _partial_ot_distance(
            valid_keywords[index], valid_patches[negative_image], partial_ratio,
            epsilon, sinkhorn_iterations)
        image_to_text_negative = _partial_ot_distance(
            valid_keywords[negative_text], valid_patches[index], partial_ratio,
            epsilon, sinkhorn_iterations)
        losses.append(0.5 * (
            F.relu(margin + positive - text_to_image_negative) +
            F.relu(margin + positive - image_to_text_negative)))

    if not losses:
        return keyword_features.sum() * 0.0
    return torch.stack(losses).mean()


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

