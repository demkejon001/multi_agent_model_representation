import torch
import torch.nn.functional as F


@torch.no_grad()
def action_acc(pred, target):
    pred_max_indices = torch.max(pred, dim=1)[1].unsqueeze(1)
    if target.dim() == 1:
        return torch.sum(target.unsqueeze(1).eq(pred_max_indices)).item() / target.shape[0]
    return torch.sum(target.eq(pred_max_indices)).item() / target.shape[0]


@torch.no_grad()
def goal_acc(pred, target):
    return torch.sum((torch.sigmoid(pred) > 0.5) == target.bool()).item() / (target.shape[0] * target.shape[1])


@torch.no_grad()
def successor_representation_acc(pred, target):
    """
    It's not the best way to measure accuracy. It lines up each 2d successor representation prediction with
    it's target and gets their absolute difference. Because the largest difference between two
    distributions is 2 (e.g. sum(abs([0,1] - [1,0])) = 2) so we divide by 2 to normalize the error.

    :param pred: shape (batch, num_discount_factors, row, col)
    :param target: shape (batch, num_discount_factors, row, col) same dimensions as pred
    """
    normalization = 2
    batch_size, num_discount_factors, row, col = target.shape
    pred_softmax = torch.softmax(pred.view((batch_size, num_discount_factors, row*col)), dim=2).view(target.shape)
    return 1 - (torch.abs(pred_softmax - target).sum() / (normalization * num_discount_factors * batch_size)).item()


def action_loss(action_hat, action_truth):
    return F.cross_entropy(action_hat, action_truth)


def goal_consumption_loss(goal_consumption_hat, goal_consumption_truth):
    device = goal_consumption_truth.device
    return F.binary_cross_entropy_with_logits(goal_consumption_hat, goal_consumption_truth,
                                              pos_weight=torch.full((goal_consumption_hat.size(1),), 5.).to(device))


def successor_representation_loss(srs_hat, srs_truth):
    """
    Cross Entropy loss across target's distribution, rather than a single target.
    Expects the srs_hat to be logits and srs_truth to be a probability distribution (where values along row X col sum to 1)

    :param srs_hat: shape (batch, num_discount_factors, row, col)
    :param srs_truth: shape (batch, num_discount_factors, row, col) same dimensions as srs_hat
    """
    batch_size, num_discount_factors, row, col = srs_hat.shape
    srs_view = (batch_size, num_discount_factors, row*col)
    return -(srs_truth.view(srs_view) * torch.log_softmax(srs_hat.view(srs_view), dim=2)).sum() / batch_size
