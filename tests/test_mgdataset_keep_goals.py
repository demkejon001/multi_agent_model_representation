import copy
from typing import List
import numpy as np

from tommas.data.gridworld_trajectory import GridworldTrajectory
from tommas.env.gridworld_env import GridworldEnv
from tommas.env.gridworld import GridWorld
from tommas.agents.reward_function import get_cooperative_reward_functions
from tommas.data.gridworld_dataset import GridworldDataset


def assert_state_correct(test_state, num_agents, num_goals, selected_agent, attach_agent_ids=False):
    starting_state = np.array(
        [[[1., 1., 1., 1., 1.],
          [1., 0., 0., 0., 1.],
          [1., 0., 0., 0., 1.],
          [1., 0., 0., 0., 1.],
          [1., 1., 1., 1., 1.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 1., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 1., 0., 0., 0.],
          [0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0.],
          [0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0.],
          [0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]],

         [[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]],

         [[2., 2., 2., 2., 2.],
          [2., 2., 2., 2., 2.],
          [2., 2., 2., 2., 2.],
          [2., 2., 2., 2., 2.],
          [2., 2., 2., 2., 2.]]]
    )
    starting_state[[-num_agents, -num_agents + selected_agent]] = starting_state[
        [-num_agents + selected_agent, -num_agents]]
    agent_offset = 1 + num_goals
    starting_state[[agent_offset, agent_offset + selected_agent]] = starting_state[
        [agent_offset + selected_agent, agent_offset]]
    if attach_agent_ids:
        assert np.array_equal(starting_state, test_state)
    else:
        assert np.array_equal(starting_state[:-num_agents], test_state)


def assert_trajectory_keep_goals(trajectory, num_goals):
    if trajectory is None:
        return

    assert np.all(np.any(trajectory[0][1:1+num_goals], axis=(1, 2)))

    if len(trajectory) >= 2:
        for i in range(1, len(trajectory)):
            assert np.array_equal(trajectory[0][:1+num_goals], trajectory[i][:1+num_goals])


def test_get_no_past_empty_current_traj():
    min_num_episodes_per_agent = 2
    num_agents = 3
    num_goals = 4
    horizon = 5
    dim = (5, 5)
    reward_funcs = get_cooperative_reward_functions(3, np.ones(num_goals))

    agent_ids = np.arange(num_agents)
    history = []
    for action in range(min_num_episodes_per_agent):
        env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
        env.reset()
        gridworld_import = [[-1, -1, -1, -1, -1],
                            [-1, 0, 2, 4, -1],
                            [-1, 1, 0, 3, -1],
                            [-1, 50, 51, 52, -1],
                            [-1, -1, -1, -1, -1]]

        gridworld = GridWorld(num_agents, num_goals, dim)
        gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
        env.world = gridworld
        env.dim = dim
        state = np.copy(gridworld.state)
        traj = []
        for _ in range(horizon):
            actions = [action for _ in range(num_agents)]
            new_state, reward, done, info = env.step(actions)
            traj.append((state, reward, actions, info))
            state = new_state
        history.append(GridworldTrajectory(traj, env, agent_ids))

    for attach_agent_ids in [True, False]:
        dataset = GridworldDataset(history)
        assert dataset.min_num_episodes_per_agent == min_num_episodes_per_agent
        assert np.array_equal(dataset.agent_ids, agent_ids)
        assert dataset.num_agents == num_agents

        n_past = 0
        empty_current_traj = True

        for agent_idx in range(num_agents):
            item = agent_idx, n_past, empty_current_traj, {"attach_agent_ids": attach_agent_ids, "keep_goals": True}

            for _ in range(5):
                past_trajectories, start_trajectory, state, action, goal_consumption, srs = dataset.__getitem__(item)

                assert past_trajectories is None
                assert start_trajectory is None
                if attach_agent_ids:
                    selected_agent = np.where(state[-num_agents:] == 0)[0][0]
                    assert agent_idx == selected_agent

                assert_state_correct(state, num_agents, num_goals, agent_idx, attach_agent_ids=attach_agent_ids)

                if action == 0:
                    true_goal_consumptions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]]
                    assert np.array_equal(goal_consumption, true_goal_consumptions[agent_idx])
                elif action == 1:
                    assert np.array_equal(goal_consumption, np.zeros(num_goals))
                up_action_srs_columns = [[0., 0.46666667, 0.53333333, 0., 0.],
                                         [0., 0.7092178, 0.2907822, 0., 0.],
                                         [0., 0.74621859, 0.25378141, 0., 0.]]
                right_action_srs_columns = [[0., 0., 0., 1., 0.],
                                            [0., 0., 0., 1., 0.],
                                            [0., 0., 0., 1., 0.]]
                true_srs = np.zeros((3, *dim))
                for srs_index, srs_column_values in enumerate(up_action_srs_columns if action == 0 else right_action_srs_columns):
                    true_srs[srs_index, :, agent_idx+1] = srs_column_values
                assert np.allclose(true_srs, srs)


def test_get_no_past():
    min_num_episodes_per_agent = 2
    num_agents = 3
    num_goals = 4
    horizon = 5
    dim = (5, 5)
    reward_funcs = get_cooperative_reward_functions(3, np.ones(num_goals))

    agent_ids = np.arange(num_agents)
    history = []
    for action in range(min_num_episodes_per_agent):
        env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
        env.reset()
        gridworld_import = [[-1, -1, -1, -1, -1],
                            [-1, 0, 2, 4, -1],
                            [-1, 1, 0, 3, -1],
                            [-1, 50, 51, 52, -1],
                            [-1, -1, -1, -1, -1]]

        gridworld = GridWorld(num_agents, num_goals, dim)
        gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
        env.world = gridworld
        env.dim = dim
        state = np.copy(gridworld.state)
        traj = []
        for _ in range(horizon):
            actions = [action for _ in range(num_agents)]
            new_state, reward, done, info = env.step(actions)
            traj.append((state, reward, actions, info))
            state = new_state
        history.append(GridworldTrajectory(traj, env, agent_ids))

    for attach_agent_ids in [True, False]:
        dataset = GridworldDataset(history)
        assert dataset.min_num_episodes_per_agent == min_num_episodes_per_agent
        assert np.array_equal(dataset.agent_ids, agent_ids)
        assert dataset.num_agents == num_agents

        n_past = 0
        empty_current_traj = False
        for agent_idx in range(num_agents):
            item = agent_idx, n_past, empty_current_traj, {"attach_agent_ids": attach_agent_ids, "keep_goals": True}

            for _ in range(10):
                past_trajectories, start_trajectory, state, action, goal_consumption, srs = dataset.__getitem__(item)

                assert past_trajectories is None

                true_keep_goals_state = np.array(
                [[[1., 1., 1., 1., 1.],
                  [1., 0., 0., 0., 1.],
                  [1., 0., 0., 0., 1.],
                  [1., 0., 0., 0., 1.],
                  [1., 1., 1., 1., 1.]],

                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 1., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],

                [[0., 0., 0., 0., 0.],
                 [0., 0., 1., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],

                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 1., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],

                 [[0., 0., 0., 0., 0.],
                  [0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]]])

                for i in range(1, len(start_trajectory)):
                    assert np.array_equal(start_trajectory[0][:1+num_goals], true_keep_goals_state)

                if attach_agent_ids:
                    assert np.array_equal(start_trajectory[0][-num_agents:], state[-num_agents:])
                    selected_agent = np.where(state[-num_agents:] == 0)[0][0]
                    assert selected_agent == agent_idx

                action_offset = 1+num_goals+num_agents
                num_actions = 5
                action_offset_end = action_offset + (num_agents*num_actions)
                state_without_actions = np.concatenate((start_trajectory[0][:action_offset], start_trajectory[0][action_offset_end:]))
                assert_state_correct(state_without_actions, num_agents, num_goals, agent_idx, attach_agent_ids)

                start_traj_action = np.where(start_trajectory[0][action_offset:action_offset+5] == 1)[0][0]
                assert start_traj_action == action

                if action == 1:
                    assert np.array_equal(goal_consumption, np.zeros(num_goals))
                    right_action_srs_columns = [[0., 0., 0., 1., 0.],
                                                [0., 0., 0., 1., 0.],
                                                [0., 0., 0., 1., 0.]]
                    true_srs = np.zeros((3, *dim))
                    for srs_index, srs_column_values in enumerate(right_action_srs_columns):
                        true_srs[srs_index, :, agent_idx + 1] = srs_column_values
                    assert np.allclose(true_srs, srs)

                else:
                    start_traj_len = start_trajectory.shape[0]
                    if start_traj_len <= 1:
                        true_goal_consumptions = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
                        assert np.array_equal(goal_consumption, true_goal_consumptions[agent_idx])
                    else:
                        assert np.array_equal(goal_consumption, np.zeros(num_goals))

                    srs_columns = [[0., 1., 0., 0., 0.],
                                   [0., 1., 0., 0., 0.],
                                   [0., 1., 0., 0., 0.]]
                    true_srs = np.zeros((3, *dim))
                    for srs_index, srs_column_values in enumerate(srs_columns):
                        true_srs[srs_index, :, agent_idx+1] = srs_column_values
                    assert np.allclose(true_srs, srs)


def test_get_with_past():
    min_num_episodes_per_agent = 2
    num_agents = 3
    num_goals = 4
    horizon = 5
    dim = (5, 5)
    reward_funcs = get_cooperative_reward_functions(3, np.ones(num_goals))

    agent_ids = np.arange(num_agents)
    history = []
    for action in range(min_num_episodes_per_agent):
        env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
        env.reset()
        gridworld_import = [[-1, -1, -1, -1, -1],
                            [-1, 0, 2, 4, -1],
                            [-1, 1, 0, 3, -1],
                            [-1, 50, 51, 52, -1],
                            [-1, -1, -1, -1, -1]]

        gridworld = GridWorld(num_agents, num_goals, dim)
        gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
        env.world = gridworld
        env.dim = dim
        state = np.copy(gridworld.state)
        traj = []
        for _ in range(horizon):
            actions = [action for _ in range(num_agents)]
            new_state, reward, done, info = env.step(actions)
            traj.append((state, reward, actions, info))
            state = new_state
        history.append(GridworldTrajectory(traj, env, agent_ids))

    for attach_agent_ids in [True, False]:
        dataset = GridworldDataset(history)
        assert dataset.min_num_episodes_per_agent == min_num_episodes_per_agent
        assert np.array_equal(dataset.agent_ids, agent_ids)
        assert dataset.num_agents == num_agents

        n_past = 1
        empty_current_traj = False

        action_offset = 1 + num_goals + num_agents
        num_actions = 5
        action_offset_end = action_offset + (num_agents * num_actions)
        for agent_idx in range(num_agents):
            item = agent_idx, n_past, empty_current_traj, {"attach_agent_ids": attach_agent_ids, "keep_goals": True}

            for _ in range(10):
                past_trajectories, start_trajectory, state, action, goal_consumption, srs = dataset.__getitem__(item)

                for traj in past_trajectories:
                    assert_trajectory_keep_goals(traj, num_goals)
                assert_trajectory_keep_goals(start_trajectory, num_goals)

                if attach_agent_ids:
                    assert np.array_equal(past_trajectories[0][0][-num_agents:], state[-num_agents:])
                    assert np.array_equal(start_trajectory[0][-num_agents:], state[-num_agents:])
                    selected_agent = np.where(state[-num_agents:] == 0)[0][0]
                    assert selected_agent == agent_idx

                spatialized_past_action = past_trajectories[0][0][action_offset:action_offset + 5]
                past_action = np.where(spatialized_past_action == 1)[0][0]
                for seq_idx in range(1, past_trajectories[0].shape[0]):
                    assert np.array_equal(spatialized_past_action, past_trajectories[0][seq_idx][action_offset:action_offset + 5])

                if past_action == 0:
                    assert action == 1
                else:
                    assert action == 0

                start_traj_action = np.where(start_trajectory[0][action_offset:action_offset+5] == 1)[0][0]
                assert start_traj_action == action

                state_without_actions = np.concatenate(
                    (start_trajectory[0][:action_offset], start_trajectory[0][action_offset_end:]))
                assert_state_correct(state_without_actions, num_agents, num_goals, agent_idx, attach_agent_ids)

                if action == 1:
                    assert np.array_equal(goal_consumption, np.zeros(num_goals))
                    right_action_srs_columns = [[0., 0., 0., 1., 0.],
                                                [0., 0., 0., 1., 0.],
                                                [0., 0., 0., 1., 0.]]
                    true_srs = np.zeros((3, *dim))
                    for srs_index, srs_column_values in enumerate(right_action_srs_columns):
                        true_srs[srs_index, :, agent_idx + 1] = srs_column_values
                    assert np.allclose(true_srs, srs)

                else:
                    start_traj_len = start_trajectory.shape[0]
                    if start_traj_len <= 1:
                        true_goal_consumptions = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
                        assert np.array_equal(goal_consumption, true_goal_consumptions[agent_idx])
                    else:
                        assert np.array_equal(goal_consumption, np.zeros(num_goals))

                    srs_columns = [[0., 1., 0., 0., 0.],
                                   [0., 1., 0., 0., 0.],
                                   [0., 1., 0., 0., 0.]]

                    true_srs = np.zeros((3, *dim))
                    for srs_index, srs_column_values in enumerate(srs_columns):
                        true_srs[srs_index, :, agent_idx + 1] = srs_column_values
                    assert np.allclose(true_srs, srs)

