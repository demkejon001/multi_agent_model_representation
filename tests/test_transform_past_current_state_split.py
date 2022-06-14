import numpy as np

from tommas.data.gridworld_transforms import PastCurrentStateSplit
from tests.fixtures import trajectory_fetcher, agent_objectives, num_agents, num_goals, dim, horizon, num_actions


def assert_state_correct(state, trajectory, step, selected_agent, attach_agent_ids):
    def add_agent_ids():
        ids = np.arange(num_agents)
        ids[[0, selected_agent]] = ids[[selected_agent, 0]]
        spatial_ids = np.full((num_agents, *dim), np.expand_dims(ids, axis=(1, 2)))
        return np.concatenate((true_state, spatial_ids), axis=0)

    true_state = trajectory[step][:-num_actions * num_agents].copy()
    true_state[[-num_agents, -num_agents + selected_agent]] = true_state[
        [-num_agents + selected_agent, -num_agents]]
    if attach_agent_ids:
        true_state = add_agent_ids()

    assert np.array_equal(true_state, state)


def assert_trajectory_correct(traj, true_traj, step, selected_agent, attach_agent_ids):
    def add_agent_ids():
        ids = np.arange(num_agents)
        ids[[0, selected_agent]] = ids[[selected_agent, 0]]
        spatial_ids = np.full((num_agents, *dim), np.expand_dims(ids, axis=(1, 2)))
        return np.concatenate((state, spatial_ids), axis=0)
    if step == 0:
        assert traj is None
        return

    action_offset = 1 + num_goals + num_agents
    agent0_actions = list(range(action_offset, action_offset + 5))
    selected_agent_actions = list(range(action_offset + (5 * selected_agent), action_offset + 5 + (5 * selected_agent)))
    agent0_position = [1 + num_goals]
    selected_agent_position = [1 + num_goals + selected_agent]

    true_traj_copy = true_traj[:step].copy()
    adjusted_traj = []
    for i, state in enumerate(true_traj_copy):
        state[agent0_position + selected_agent_position] = state[selected_agent_position + agent0_position]
        state[agent0_actions + selected_agent_actions] = state[selected_agent_actions + agent0_actions]
        if attach_agent_ids:
            state = add_agent_ids()
        adjusted_traj.append(state)

    assert np.array_equal(adjusted_traj, traj)


def assert_ia_features_correct(ia_features):
    action_offset = 1 + num_goals + num_agents
    true_ia_features = list(range(1 + num_goals + 1)) + list(range(action_offset, action_offset + num_actions))
    assert np.array_equal(ia_features, true_ia_features)


def assert_agent_objectives(action, goal_consumption, srs, true_objectives):
    return (np.array_equal(action, true_objectives[0]) and
            np.array_equal(goal_consumption, true_objectives[1]) and
            np.allclose(srs, true_objectives[2])
            )


def test_get_no_past_traj(trajectory_fetcher, agent_objectives):
    true_actions, true_goals_consumed, true_srs = agent_objectives
    for attach_agent_ids in [True, False]:
        gridworld_transform = PastCurrentStateSplit(attach_agent_ids=attach_agent_ids)
        n_past = 0
        for agent_id in range(num_agents):
            item = agent_id, n_past
            for _ in range(5):
                trajectories, agent_idx = trajectory_fetcher.__getitem__(item)
                current_spatialised_trajectory = trajectories[0].get_spatialised_trajectory()
                for step in range(horizon):
                    batch = gridworld_transform(trajectories, agent_idx, n_past, current_traj_len=step)
                    past_trajectories, start_trajectory, state, action, goal_consumption, srs, ia_features = batch
                    assert_ia_features_correct(ia_features)
                    assert past_trajectories is None

                    assert_state_correct(state, current_spatialised_trajectory, step, agent_idx, attach_agent_ids)
                    assert_trajectory_correct(start_trajectory, current_spatialised_trajectory, step, agent_idx, attach_agent_ids)
                    traj1_true_objectives = [
                        true_actions[agent_idx][0][step],
                        true_goals_consumed[agent_idx][0][step],
                        true_srs[agent_idx][0][step],
                    ]
                    traj2_true_objectives = [
                        true_actions[agent_idx][1][step],
                        true_goals_consumed[agent_idx][1][step],
                        true_srs[agent_idx][1][step],
                    ]

                    assert (assert_agent_objectives(action, goal_consumption, srs, traj1_true_objectives) or
                            assert_agent_objectives(action, goal_consumption, srs, traj2_true_objectives))


def test_with_past_traj(trajectory_fetcher, agent_objectives):
    true_actions, true_goals_consumed, true_srs = agent_objectives
    for attach_agent_ids in [True, False]:
        gridworld_transform = PastCurrentStateSplit(attach_agent_ids=attach_agent_ids)
        n_past = 1
        for agent_id in range(num_agents):
            item = agent_id, n_past
            for _ in range(5):
                trajectories, agent_idx = trajectory_fetcher.__getitem__(item)
                current_spatialised_trajectory = trajectories[0].get_spatialised_trajectory()
                past_spatialised_trajectory = trajectories[1].get_spatialised_trajectory(agent_idx=agent_idx,
                                                                                         attach_agent_ids=attach_agent_ids)
                for step in range(horizon):
                    batch = gridworld_transform(trajectories, agent_idx, n_past, current_traj_len=step)
                    past_trajectories, start_trajectory, state, action, goal_consumption, srs, ia_features = batch

                    assert np.array_equal(np.expand_dims(past_spatialised_trajectory, axis=0), past_trajectories)
                    assert_ia_features_correct(ia_features)

                    assert_state_correct(state, current_spatialised_trajectory, step, agent_idx, attach_agent_ids)
                    assert_trajectory_correct(start_trajectory, current_spatialised_trajectory, step, agent_idx, attach_agent_ids)
                    traj1_true_objectives = [
                        true_actions[agent_idx][0][step],
                        true_goals_consumed[agent_idx][0][step],
                        true_srs[agent_idx][0][step],
                    ]
                    traj2_true_objectives = [
                        true_actions[agent_idx][1][step],
                        true_goals_consumed[agent_idx][1][step],
                        true_srs[agent_idx][1][step],
                    ]

                    assert (assert_agent_objectives(action, goal_consumption, srs, traj1_true_objectives) or
                            assert_agent_objectives(action, goal_consumption, srs, traj2_true_objectives))
