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


class WeightSoftCELoss(nn.Module):
    """Weighted soft-target cross entropy used as TC loss in LSCI-TC."""

    def __init__(self, imagegamma=2.0, textgamma=2.0, alpha=0.1,
                 maxgamma=5.0, mingamma=-1.0):
        super().__init__()
        self.imagegamma = float(imagegamma)
        self.textgamma = float(textgamma)
        self.alpha = float(alpha)
        self.register_buffer('maxgamma', torch.tensor(float(maxgamma)))
        self.register_buffer('mingamma', torch.tensor(float(mingamma)))
        self.register_buffer('image_to_textgamma', torch.tensor(self.imagegamma))
        self.register_buffer('text_to_imagegamma', torch.tensor(self.textgamma))

    def update_gamma(self, which, gap, max_change=0.2):
        gamma = self.image_to_textgamma if which == 'i2t' else self.text_to_imagegamma
        gap = torch.as_tensor(gap, dtype=gamma.dtype, device=gamma.device)
        exp_factor = torch.tanh(gap) + 1.0
        target = torch.clamp(gamma * exp_factor, self.mingamma, self.maxgamma)
        delta = torch.clamp(target - gamma, -max_change, max_change)
        new_gamma = gamma + delta
        base_gamma = self.imagegamma if which == 'i2t' else self.textgamma
        if abs(base_gamma) < abs(self.alpha * self.imagegamma):
            new_gamma = gamma.new_tensor(-self.alpha * self.imagegamma)
        gamma.copy_(new_gamma.reshape_as(gamma))

    def updategamma(self, image_text_meancalibration_gap, text_image_meancalibration_gap):
        self.update_gamma('i2t', image_text_meancalibration_gap)
        self.update_gamma('t2i', text_image_meancalibration_gap)
        if abs(float(self.image_to_textgamma)) < abs(self.alpha * self.imagegamma):
            self.image_to_textgamma.fill_(-self.alpha * self.imagegamma)
        elif abs(float(self.text_to_imagegamma)) < abs(self.alpha * self.textgamma):
            self.text_to_imagegamma.fill_(-self.alpha * self.textgamma)

    def forward(self, inputs, labels, text_to_image=False):
        log_inputs = F.log_softmax(inputs, dim=1)
        pt = log_inputs.exp()
        if text_to_image:
            gamma = self.text_to_imagegamma
        else:
            gamma = self.image_to_textgamma
        gamma = gamma.to(device=inputs.device, dtype=inputs.dtype)
        gamma_sign = torch.sign(gamma)
        pt = gamma_sign * pt
        gamma_mag = gamma.abs().detach()
        weight_targets = torch.clamp(torch.abs(labels - pt), min=1e-5, max=2).pow(gamma_mag) * labels
        loss = torch._C._nn.cross_entropy_loss(inputs, weight_targets).mean()
        return loss


def hard_pid_targets(pid):
    """Row-normalized multi-positive targets, same matching rule as SDM."""
    pid = pid.reshape(-1, 1)
    match = (pid == pid.t()).float()
    return match / match.sum(dim=1, keepdim=True)


def compute_tc(image_features, text_features, teacher_image, teacher_text, pid,
               logit_scale, teacher_temperature, soft_weight, criterion):
    """
    TC loss.

    Hard targets are the row-normalized same-image caption matches.
    The soft component is the momentum teacher's similarity, with the
    teacher stopped from gradients. Student logits keep IRRA's logit_scale.
    """
    hard = hard_pid_targets(pid).to(image_features.device)
    soft_weight = image_features.new_tensor(float(soft_weight))

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scale = logit_scale.to(image_features.device)
    logits_per_image = scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    teacher_image = teacher_image / teacher_image.norm(dim=-1, keepdim=True)
    teacher_text = teacher_text / teacher_text.norm(dim=-1, keepdim=True)
    teacher_scale = image_features.new_tensor(1.0 / float(teacher_temperature))
    soft_i2t = F.softmax(teacher_scale * teacher_image @ teacher_text.t(), dim=1)
    soft_t2i = F.softmax(teacher_scale * teacher_text @ teacher_image.t(), dim=1)

    targets_i2t = (1.0 - soft_weight) * hard + soft_weight * soft_i2t
    targets_t2i = (1.0 - soft_weight) * hard + soft_weight * soft_t2i
    loss_i = criterion(logits_per_image, targets_i2t, text_to_image=False)
    loss_t = criterion(logits_per_text, targets_t2i, text_to_image=True)
    return (loss_i + loss_t) / 2

