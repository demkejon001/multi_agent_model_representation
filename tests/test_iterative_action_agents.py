import math
import numpy as np
from typing import List

from tommas.agents.iterative_action_agents import PureStrategyAgent, MixedStrategyAgent, ActionPatternAgent, \
    TriggerPatternAgent, GrimTriggerAgent, MirrorAgent, WSLSAgent, ActionPatternAndMixedStrategyAgent, \
    ActionPatternAndMirrorAgent, IterativeActionAgent, MixedTriggerPattern


def get_random_state():
    return list(np.random.randint(0, 2, size=(1, 2)))


def get_agent_traj(agent: IterativeActionAgent, opponent_traj: List[int]):
    agent.reset()
    state = [[-1, -1]]
    agent_actions = []
    for opponent_action in opponent_traj:
        agent_action = agent.action(state)
        agent_actions.append(agent_action)
        state.append([agent_action, opponent_action])
    return agent_actions


def test_pure_strategy_agent():
    for true_action in [0, 1]:
        agent = PureStrategyAgent(0, 0, true_action)
        for _ in range(10):
            agent_action = agent.action(get_random_state())
            assert agent_action == true_action
        agent.reset()
        for _ in range(10):
            agent_action = agent.action(get_random_state())
            assert agent_action == true_action


def test_mixed_strategy_agent():
    def assert_agent_action_sample():
        actions = [agent.action(get_random_state()) for _ in range(num_samples)]
        _, counts = np.unique(actions, return_counts=True)
        assert (num_samples * .2) < counts[0] < (num_samples * .8)

    def assert_agent_pure_action(pure_action):
        actions = [agent.action(get_random_state()) for _ in range(20)]
        unique_actions = np.unique(actions)
        assert len(unique_actions) == 1 and unique_actions[0] == pure_action

    num_samples = 200
    agent = MixedStrategyAgent(0, 0, [.5, .5])
    assert_agent_action_sample()
    agent.reset()
    assert_agent_action_sample()

    agent = MixedStrategyAgent(0, 0, [1., 0])
    assert_agent_pure_action(0)
    agent.reset()
    assert_agent_pure_action(0)

    agent = MixedStrategyAgent(0, 0, [0, 1.])
    assert_agent_pure_action(1)
    agent.reset()
    assert_agent_pure_action(1)


def test_action_pattern_agent():
    patterns = [[0, 1], [0, 1, 0], [1, 1, 0, 0, 1]]
    for pattern in patterns:
        horizon = 31
        true_action_traj = np.tile(pattern, math.ceil(horizon / len(pattern)))[:horizon]
        agent = ActionPatternAgent(0, 0, pattern)
        action_traj = [agent.action(get_random_state()) for _ in range(horizon)]
        assert np.array_equal(action_traj, true_action_traj)
        agent.reset()
        action_traj = [agent.action(get_random_state()) for _ in range(horizon)]
        assert np.array_equal(action_traj, true_action_traj)


def test_grim_trigger():
    opponent_trajs = [[0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1, 1]]
    for starting_action in [0, 1]:
        for trigger_action in [0, 1]:
            agent = GrimTriggerAgent(0, 0, starting_action, trigger_action)
            for opponent_traj in opponent_trajs:
                state = [[-1, -1]]
                agent_actions = []
                for opponent_action in opponent_traj:
                    agent_action = agent.action(state)
                    agent_actions.append(agent_action)
                    state.append([agent_action, opponent_action])
                agent.reset()
                trigger_index = opponent_traj.index(trigger_action)
                true_action_traj = [starting_action if i <= trigger_index else (1 - starting_action) for i in range(len(opponent_traj))]
                assert np.array_equal(true_action_traj, agent_actions)


def test_mirror_agent():
    opponent_trajs = [[0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1, 1]]
    for starting_action in [0, 1]:
        agent = MirrorAgent(0, 0, starting_action)
        for opponent_traj in opponent_trajs:
            state = [[-1, -1]]
            agent_actions = []
            for opponent_action in opponent_traj:
                agent_action = agent.action(state)
                agent_actions.append(agent_action)
                state.append([agent_action, opponent_action])
            agent.reset()
            true_agent_actions = [starting_action] + opponent_traj[:-1]
            assert np.array_equal(agent_actions, true_agent_actions)


def test_wsls_agent():
    opponent_trajs = [[0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1, 1]]
    for starting_action in [0, 1]:
        for trigger_action in [0, 1]:
            agent = WSLSAgent(0, 0, starting_action, trigger_action)
            for opponent_traj in opponent_trajs:
                state = [[-1, -1]]
                agent_actions = []
                true_agent_actions = [starting_action]
                for opponent_action in opponent_traj:
                    if opponent_action == trigger_action:
                        true_agent_actions.append(1 - true_agent_actions[-1])
                    else:
                        true_agent_actions.append(true_agent_actions[-1])
                    agent_action = agent.action(state)
                    agent_actions.append(agent_action)
                    state.append([agent_action, opponent_action])
                true_agent_actions = true_agent_actions[:-1]
                agent.reset()
                assert np.array_equal(agent_actions, true_agent_actions)


def test_trigger_pattern_agent():
    def assert_agent_actions():
        agent_actions = get_agent_traj(agent, opponent_traj)
        assert np.array_equal(agent_actions, true_actions)

    agent = TriggerPatternAgent(0, 0, starting_action=0, trigger_action=1, trigger_patience=2, action_pattern=[0, 1, 0])
    opponent_traj = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    true_actions = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert_agent_actions()

    opponent_traj = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
    true_actions = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    assert_agent_actions()

    opponent_traj = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    true_actions = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert_agent_actions()

    agent = TriggerPatternAgent(0, 0, starting_action=1, trigger_action=1, trigger_patience=1, action_pattern=[0, 1])
    opponent_traj = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    true_actions = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]
    assert_agent_actions()

    opponent_traj = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
    true_actions = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    assert_agent_actions()

    agent = TriggerPatternAgent(0, 0, starting_action=1, trigger_action=0, trigger_patience=3, action_pattern=[0])
    opponent_traj = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    true_actions = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
    assert_agent_actions()

    opponent_traj = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
    true_actions = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    assert_agent_actions()

    agent = TriggerPatternAgent(0, 0, starting_action=1, trigger_action=[1, 0], trigger_patience=1,
                                action_pattern=[1, 0, 0])
    opponent_traj = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
    true_actions = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    assert_agent_actions()

    opponent_traj = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    true_actions = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    assert_agent_actions()

    agent = TriggerPatternAgent(0, 0, starting_action=1, trigger_action=[1, 0], trigger_patience=2,
                                action_pattern=[1, 0, 0])
    opponent_traj = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
    true_actions = [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert_agent_actions()

    opponent_traj = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    true_actions = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    assert_agent_actions()


def test_action_pattern_and_mixed_strategy_agent():
    def assert_agent_actions():
        agent_actions = np.array(get_agent_traj(agent, opponent_traj))
        np_true_actions = np.array(true_actions)
        agent_actions[np_true_actions == -1] = -1
        assert np.array_equal(agent_actions, np_true_actions)

    agent = ActionPatternAndMixedStrategyAgent(0, 0, [0, 1, 0], mixed_strategy=[.7, .3])
    opponent_traj = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    true_actions = [0, 1, 0, -1, -1, -1, 0, 1, 0, -1, -1, -1]
    assert_agent_actions()

    opponent_traj = [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]
    true_actions = [0, 1, 0, -1, -1, -1, 0, 1, 0, -1, -1, -1]
    assert_agent_actions()

    agent = ActionPatternAndMixedStrategyAgent(0, 0, [1, 1, 0, 0], mixed_strategy=[1., 0.])
    opponent_traj = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
    true_actions = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    agent_traj = get_agent_traj(agent, opponent_traj)
    assert np.array_equal(agent_traj, true_actions)

    opponent_traj = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    true_actions = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    agent_traj = get_agent_traj(agent, opponent_traj)
    assert np.array_equal(agent_traj, true_actions)


def test_action_pattern_and_mirror_agent():
    def assert_agent_actions():
        agent_actions = get_agent_traj(agent, opponent_traj)
        assert np.array_equal(agent_actions, true_actions)

    agent = ActionPatternAndMirrorAgent(0, 0, [0, 1, 0])
    opponent_traj = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    true_actions = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1]
    assert_agent_actions()

    opponent_traj = [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]
    true_actions = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    assert_agent_actions()

    agent = ActionPatternAndMirrorAgent(0, 0, [1, 1, 0, 0])
    opponent_traj = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
    true_actions = [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
    assert_agent_actions()

    opponent_traj = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    true_actions = [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    assert_agent_actions()


def test_mixed_trigger_pattern_agent():
    def assert_agent_actions():
        agent_actions = get_agent_traj(agent, opponent_traj)
        assert np.array_equal(agent_actions, true_actions)

    def fuzzy_assert_agent_actions():
        agent_actions = np.array(get_agent_traj(agent, opponent_traj))
        np_true_actions = np.array(true_actions)
        agent_actions[np_true_actions == -1] = -1
        assert np.array_equal(agent_actions, np_true_actions)


    agent = MixedTriggerPattern(0, 0, mixed_strategy=[1., 0.], trigger_action=1, trigger_patience=2,
                                action_pattern=[0, 1, 0])
    opponent_traj = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    true_actions = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert_agent_actions()

    opponent_traj = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
    true_actions = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    assert_agent_actions()

    agent = MixedTriggerPattern(0, 0, mixed_strategy=[0., 1.], trigger_action=1, trigger_patience=2,
                                action_pattern=[0, 1, 0])
    opponent_traj = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    true_actions = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]
    assert_agent_actions()

    opponent_traj = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
    true_actions = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    assert_agent_actions()

    agent = MixedTriggerPattern(0, 0, mixed_strategy=[.5, .5], trigger_action=[1, 0], trigger_patience=1,
                                action_pattern=[1, 0, 0])
    opponent_traj = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
    true_actions = [-1, -1, -1, 1, 0, 0, -1, -1, -1, 1, 0, 0, -1, -1, -1, 1]
    fuzzy_assert_agent_actions()

    opponent_traj = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    true_actions = [-1, -1, 1, 0, 0, -1, -1, -1, 1, 0, 0, -1, -1, -1, 1, 0]
    fuzzy_assert_agent_actions()

