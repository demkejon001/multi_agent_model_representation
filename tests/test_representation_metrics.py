import numpy as np
from tommas.analysis.representation_metrics import generate_random_traj


def test_generate_random_traj_reproducible():
    traj1 = generate_random_traj("wsls", {}, seed=4, n_past=2, opponent_idx=1)
    traj2 = generate_random_traj("wsls", {}, seed=4, n_past=2, opponent_idx=1)
    assert np.array_equal(traj1, traj2)

    traj1 = generate_random_traj("wsls", {}, seed=6, n_past=2, opponent_idx=1)
    traj2 = generate_random_traj("wsls", {}, seed=6, n_past=2, opponent_idx=1)
    assert np.array_equal(traj1, traj2)

    traj1 = generate_random_traj("wsls", {}, seed=4, n_past=2, opponent_idx=1)
    traj2 = generate_random_traj("wsls", {}, seed=6, n_past=2, opponent_idx=1)
    assert not np.array_equal(traj1, traj2)


def test_generate_random_traj_reproducible_opponent_idx():
    traj1 = generate_random_traj("wsls", {}, seed=4, n_past=2, num_agents=4, opponent_idx=1)
    traj2 = generate_random_traj("wsls", {}, seed=4, n_past=2, num_agents=4, opponent_idx=2)
    traj3 = generate_random_traj("wsls", {}, seed=4, n_past=2, num_agents=4, opponent_idx=3)

    traj2 = traj2[:, :, [0, 2, 1, 3]]
    traj3 = traj3[:, :, [0, 3, 2, 1]]

    assert np.array_equal(traj1, traj2)
    assert np.array_equal(traj1, traj3)
