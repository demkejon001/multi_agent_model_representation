import pytest
import torch
from tommas.helper_code.metrics import action_transformer_loss, goal_consumption_transformer_loss, action_acc, \
    goal_acc, successor_representation_loss, successor_representation_acc, successor_representation_transformer_loss, \
    action_loss, goal_consumption_loss


def test_action_acc():
    pred = torch.tensor([[.3, .4, .3], [.1, -.1, 4], [2, 4, 23]])
    target = torch.tensor([[1], [2], [0]])
    assert action_acc(pred, target) == pytest.approx(2. / 3.)


def test_goal_acc():
    pred = torch.tensor([[-.1, .1, -.3], [-.1, .2, 4], [2, 4, 3]])
    target = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 0, 1]])
    assert goal_acc(pred, target) == pytest.approx(6. / 9.)


def test_srs_acc():
    """
    softmax(pred) = [[[[.25, .25], [.25, .25]], [[.7, .1], [.1, .1]], [[.1, .3], [.3, .3]]],
                     [[[.25, .25], [.25, .25]], [[.7, .1], [.1, .1]], [[.1, .3], [.3, .3]]]]
    1.5 = abs(.25 - .1) + abs(.25 - .3) + abs(.25 - .3) + abs(.25 - .3) + abs(.7 - .1) + abs(.1 - .3) + abs(.1 - .3)
                        + abs(.1 - .3) + 0
    1.2 = 0 + abs(.7 - .25) + abs(.1 - .25) + abs(.1 - .25) + abs(.1 - .25) + abs(.1 - .25) + abs(.3 - .25)
            + abs(.3 - .25) + abs(.3 - .25)

    .775 = 1 - (1.5 + 1.2 / (3 * 2 * 2))  # Num_sr_discount_factors = 3, normalization = 2, batch = 2
    """
    pred = torch.tensor([[[[-1.386294, -1.386294], [-1.386294, -1.386294]],
                          [[-0.356675, -2.302585], [-2.302585, -2.302585]],
                          [[-2.302585, -1.203973], [-1.203973, -1.203973]]],
                         [[[-1.386294, -1.386294], [-1.386294, -1.386294]],
                          [[-0.356675, -2.302585], [-2.302585, -2.302585]],
                          [[-2.302585, -1.203973], [-1.203973, -1.203973]]]])
    target = torch.tensor([[[[.1, .3], [.3, .3]], [[.1, .3], [.3, .3]], [[.1, .3], [.3, .3]]],
                           [[[.25, .25], [.25, .25]], [[.25, .25], [.25, .25]], [[.25, .25], [.25, .25]]]])
    assert successor_representation_acc(pred, target) == pytest.approx(.775)


def test_srs_loss():
    """
    softmax(pred) = [[[0.250000, 0.250000], [0.250000, 0.250000]], [[0.890372, 0.048991], [0.016308, 0.044329]], [[0.172659, 0.230746], [0.176147, 0.420447]],
                     [[0.250000, 0.250000], [0.250000, 0.250000]], [[0.377867, 0.207378], [0.207378, 0.207378]], [[0.214399, 0.261867], [0.261867, 0.261867]]]

    5.86884 = -(  ln(.25) * .1 + ln(.25) * .3 + ln(.25) * .3 + ln(.25) * .3
                + ln(.890372) * .1 + ln(.048991) * .3 + ln(.016308) * .3 + ln(.044329) * .3
                + ln(.172659) * .1 + ln(.230746) * .3 + ln(.176147) * .3 + ln(.420447) * .3  )
    4.19942 = -(  ln(.25) * .25 + ln(.25) * .25 + ln(.25) * .25 + ln(.25) * .25
                + ln(.377867) * .25 + ln(.207378) * .25 + ln(.207378) * .25 + ln(.207378) * .25
                + ln(.214399) * .25 + ln(.261867) * .25 + ln(.261867) * .25 + ln(.261867) * .25  )

    5.03413 = (5.86884 + 4.19942) / 2  # batch = 2

    """
    pred = torch.tensor([[[[.1, .1], [.1, .1]], [[3., .1], [-1., 0.]], [[.01, .3], [.03, .9]]],
                         [[[.25, .25], [.25, .25]], [[.7, .1], [.1, .1]], [[.1, .3], [.3, .3]]]])
    target = torch.tensor([[[[.1, .3], [.3, .3]], [[.1, .3], [.3, .3]], [[.1, .3], [.3, .3]]],
                           [[[.25, .25], [.25, .25]], [[.25, .25], [.25, .25]], [[.25, .25], [.25, .25]]]])
    assert successor_representation_loss(pred, target).item() == pytest.approx(5.03413, rel=.00001)


def test_action_transformer_loss():
    pred = torch.tensor([[.3, .4, .3], [.1, -.1, 4], [2, 4, 23]])
    target = torch.tensor([1, 2, 0])
    batch_size = target.size(0)
    weight = torch.ones(batch_size) / batch_size
    assert torch.allclose(action_loss(pred, target), action_transformer_loss(pred, (target, weight)))


def test_goal_transformer_loss():
    pred = torch.tensor([[-.1, .1, -.3], [-.1, .2, 4], [2, 4, 3]])
    target = torch.tensor([[0., 1, 0], [1, 1, 1], [0, 0, 1]])
    batch_size, num_goals = target.shape
    weight = torch.ones(batch_size) / (batch_size * num_goals)
    assert torch.allclose(goal_consumption_loss(pred, target), goal_consumption_transformer_loss(pred, (target, weight)))


def test_srs_transformer_loss():
    """
    softmax(pred) = [[[0.250000, 0.250000], [0.250000, 0.250000]], [[0.890372, 0.048991], [0.016308, 0.044329]], [[0.172659, 0.230746], [0.176147, 0.420447]],
                     [[0.250000, 0.250000], [0.250000, 0.250000]], [[0.377867, 0.207378], [0.207378, 0.207378]], [[0.214399, 0.261867], [0.261867, 0.261867]]]

    5.86884 = -(  ln(.25) * .1 + ln(.25) * .3 + ln(.25) * .3 + ln(.25) * .3
                + ln(.890372) * .1 + ln(.048991) * .3 + ln(.016308) * .3 + ln(.044329) * .3
                + ln(.172659) * .1 + ln(.230746) * .3 + ln(.176147) * .3 + ln(.420447) * .3  )
    4.19942 = -(  ln(.25) * .25 + ln(.25) * .25 + ln(.25) * .25 + ln(.25) * .25
                + ln(.377867) * .25 + ln(.207378) * .25 + ln(.207378) * .25 + ln(.207378) * .25
                + ln(.214399) * .25 + ln(.261867) * .25 + ln(.261867) * .25 + ln(.261867) * .25  )

    5.03413 = (5.86884 + 4.19942) / 2  # batch = 2

    """
    pred = torch.tensor([[[[.1, .1], [.1, .1]], [[3., .1], [-1., 0.]], [[.01, .3], [.03, .9]]],
                         [[[.25, .25], [.25, .25]], [[.7, .1], [.1, .1]], [[.1, .3], [.3, .3]]]])
    target = torch.tensor([[[[.1, .3], [.3, .3]], [[.1, .3], [.3, .3]], [[.1, .3], [.3, .3]]],
                           [[[.25, .25], [.25, .25]], [[.25, .25], [.25, .25]], [[.25, .25], [.25, .25]]]])

    weight = torch.tensor([.5, .5])
    assert successor_representation_transformer_loss(pred, (target, weight)).item() == pytest.approx(5.03413, rel=.00001)

