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


def compute_ot(matcher, image_global, text_global, image_tokens, text_tokens,
               caption_ids, pid, logit_scale, num_hard_negatives=3):
    """Fine-grained loss on top of the global cosine.

    The diagonal and the top hard negatives in each direction, after same-pid
    pairs are removed, become global cosine plus a gated transport score.
    Every other entry stays the global cosine. Other same-pid entries are
    ignored by the softmax.
    """
    batch_size = image_global.shape[0]
    device = image_global.device
    image_norm = F.normalize(image_global, p=2, dim=-1)
    text_norm = F.normalize(text_global, p=2, dim=-1)
    sim = image_norm @ text_norm.t()
    gate = matcher.gate().to(dtype=sim.dtype)

    pid = pid.reshape(-1)
    same_pid = pid.unsqueeze(1).eq(pid.unsqueeze(0))
    with torch.no_grad():
        mined = sim.detach().masked_fill(same_pid, float('-inf'))
        num_hard = min(int(num_hard_negatives), max(batch_size - 1, 0))
        if num_hard > 0:
            i2t_values, i2t_index = mined.topk(num_hard, dim=-1)
            t2i_values, t2i_index = mined.topk(num_hard, dim=0)
            t2i_index = t2i_index.t()
            t2i_values = t2i_values.t()
            row_valid = torch.isfinite(i2t_values)
            col_valid = torch.isfinite(t2i_values)
        else:
            i2t_index = t2i_index = None
            row_valid = col_valid = None

    lengths = caption_ids.argmax(dim=-1) + 1
    index = torch.arange(batch_size, device=device)
    pos_sim = matcher(image_norm, image_tokens, text_norm, text_tokens, lengths)
    hybrid = sim.index_put((index, index), sim[index, index] + gate * pos_sim)

    for neg in range(num_hard):
        text_index = i2t_index[:, neg]
        valid = row_valid[:, neg]
        ot_score = matcher(
            image_norm, image_tokens,
            text_norm[text_index], text_tokens[text_index], lengths[text_index])
        if bool(valid.any()):
            rows = index[valid]
            cols = text_index[valid]
            hybrid = hybrid.index_put((rows, cols), sim[rows, cols] + gate * ot_score[valid])

        image_index = t2i_index[:, neg]
        valid = col_valid[:, neg]
        ot_score = matcher(
            image_norm[image_index], image_tokens[image_index],
            text_norm, text_tokens, lengths)
        if bool(valid.any()):
            rows = image_index[valid]
            cols = index[valid]
            hybrid = hybrid.index_put((rows, cols), sim[rows, cols] + gate * ot_score[valid])

    off_pid = same_pid & ~torch.eye(batch_size, dtype=torch.bool, device=device)
    hybrid = hybrid.masked_fill(off_pid, float('-inf'))

    scale = logit_scale.to(device=device, dtype=hybrid.dtype)
    logits_per_image = hybrid * scale
    logits_per_text = logits_per_image.t()
    labels = torch.arange(batch_size, device=device)
    loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
    return loss

