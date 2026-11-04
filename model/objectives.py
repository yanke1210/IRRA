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


class TCGamma(nn.Module):
    """Persistent TC gamma and difficulty EMA. Forward only reads them."""

    def __init__(self, image_gamma=0.0, text_gamma=0.0, tau=0.05, ema=0.9,
                 step=0.2, ref=0.5, max_focal=5.0, max_inv=1.0):
        super().__init__()
        self.tau = float(tau)
        self.ema = float(ema)
        self.step = float(step)
        self.ref = float(ref)
        self.register_buffer('image_to_textgamma', torch.tensor(float(image_gamma)))
        self.register_buffer('text_to_imagegamma', torch.tensor(float(text_gamma)))
        self.register_buffer('i2t_difficulty_ema', torch.tensor(0.5))
        self.register_buffer('t2i_difficulty_ema', torch.tensor(0.5))
        self.register_buffer('last_i2t_difficulty', torch.tensor(0.5))
        self.register_buffer('last_t2i_difficulty', torch.tensor(0.5))
        self.register_buffer('max_focal', torch.tensor(float(max_focal)))
        self.register_buffer('max_inv', torch.tensor(float(max_inv)))
        self.register_buffer('i2t_diff_sum', torch.zeros(()))
        self.register_buffer('t2i_diff_sum', torch.zeros(()))
        self.register_buffer('diff_count', torch.zeros(()))

    def reset_epoch_stats(self):
        self.i2t_diff_sum.zero_()
        self.t2i_diff_sum.zero_()
        self.diff_count.zero_()

    def accumulate(self, i2t_sum, t2i_sum, count):
        self.i2t_diff_sum.add_(i2t_sum.detach())
        self.t2i_diff_sum.add_(t2i_sum.detach())
        self.diff_count.add_(count.detach())

    @torch.no_grad()
    def update_from_epoch(self, i2t_sum, t2i_sum, count):
        count = torch.as_tensor(count, dtype=self.diff_count.dtype, device=self.diff_count.device)
        if float(count) <= 0:
            return
        i2t_sum = torch.as_tensor(i2t_sum, dtype=self.i2t_diff_sum.dtype, device=self.i2t_diff_sum.device)
        t2i_sum = torch.as_tensor(t2i_sum, dtype=self.t2i_diff_sum.dtype, device=self.t2i_diff_sum.device)
        difficulty_i2t = i2t_sum / count
        difficulty_t2i = t2i_sum / count
        self.last_i2t_difficulty.copy_(difficulty_i2t)
        self.last_t2i_difficulty.copy_(difficulty_t2i)
        self.i2t_difficulty_ema.mul_(self.ema).add_(difficulty_i2t, alpha=1.0 - self.ema)
        self.t2i_difficulty_ema.mul_(self.ema).add_(difficulty_t2i, alpha=1.0 - self.ema)
        self._step_gamma(self.image_to_textgamma, self.i2t_difficulty_ema)
        self._step_gamma(self.text_to_imagegamma, self.t2i_difficulty_ema)

    def _step_gamma(self, gamma, difficulty):
        updated = gamma + self.step * (difficulty - self.ref)
        updated = updated.clamp(-self.max_inv, self.max_focal)
        gamma.copy_(updated.reshape_as(gamma))


def one_index_per_pid(pid):
    """One random row for each pid. Other rows of that pid stay out of TC."""
    pid = pid.reshape(-1)
    chosen = []
    for value in torch.unique(pid):
        rows = torch.nonzero(pid == value, as_tuple=False).reshape(-1)
        pick = torch.randint(rows.numel(), (1,), device=rows.device)
        chosen.append(rows[pick])
    return torch.cat(chosen)


def positive_pair_weight(positive_prob, gamma):
    """Focal when gamma >= 0, inverse-focal when gamma < 0. Mean weight is 1."""
    gamma = gamma.detach().to(device=positive_prob.device, dtype=positive_prob.dtype)
    focal = (1.0 - positive_prob).clamp_min(1e-6).pow(gamma.clamp(min=0))
    inverse = (1.0 + positive_prob).pow((-gamma).clamp(min=0))
    weight = torch.where(gamma >= 0, focal, inverse)
    return weight / weight.mean().clamp_min(1e-6)


def positive_pair_loss(logits, gamma):
    positive_prob = F.softmax(logits, dim=1).diag()
    weight = positive_pair_weight(positive_prob, gamma)
    return -(weight * positive_prob.clamp_min(1e-6).log()).mean()


def pair_difficulty(similarity, tau):
    """Mean hardness of the diagonal against the hardest off-diagonal entry."""
    if similarity.size(0) < 2:
        zero = similarity.new_zeros(())
        return zero, zero
    eye = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    positive = similarity.diag()
    hardest = similarity.masked_fill(eye, float('-inf')).max(dim=1).values
    hardness = torch.sigmoid(-(positive - hardest) / tau)
    return hardness.sum(), hardness.new_tensor(float(hardness.numel()))


def compute_tc(image_features, text_features, pid, logit_scale, state):
    """One-to-one TC loss and its matching difficulty, from one shared subset."""
    keep = one_index_per_pid(pid)
    image_features = image_features[keep]
    text_features = text_features[keep]
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = image_norm @ text_norm.t()
    logits = logit_scale.to(similarity.device) * similarity
    loss_i = positive_pair_loss(logits, state.image_to_textgamma)
    loss_t = positive_pair_loss(logits.t(), state.text_to_imagegamma)
    with torch.no_grad():
        i2t_sum, count = pair_difficulty(similarity, state.tau)
        t2i_sum, _ = pair_difficulty(similarity.t(), state.tau)
        state.accumulate(i2t_sum, t2i_sum, count)
    return (loss_i + loss_t) / 2

