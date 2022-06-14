from typing import Tuple
import numpy as np

from src.agents.iterative_action_agents import PureStrategyAgent, MixedStrategyAgent, ActionPatternAgent, \
    TriggerPatternAgent, GrimTriggerAgent, MirrorAgent, WSLSAgent, ActionPatternAndMixedStrategyAgent, \
    ActionPatternAndMirrorAgent, IterativeActionAgent, MixedTriggerPattern


class RandomStrategySampler:
    def __init__(self, seed: int, is_train_sampler: bool):
        self.rng = np.random.default_rng(seed)
        self.is_train_sampler = is_train_sampler
        self.action_patterns = self._get_random_action_patterns()
        self.trigger_action_patterns = self._get_random_trigger_action_patterns()

    def _get_random_action_patterns(self, train_split=.7, max_pattern_len=8):
        def generate_action_patterns():
            binary_strings = []

            def genbin(bs=None):
                def append(bs_i):
                    if all(bs_i) or not any(bs_i):
                        return
                    if len(bs_i) == 1:
                        return
                    binary_strings.append(bs_i)

                if bs is None:
                    bs = []
                if len(bs) == max_pattern_len:
                    return
                else:
                    bs1 = bs.copy()
                    bs1.append(0)
                    append(bs1)

                    bs2 = bs.copy()
                    bs2.append(1)
                    append(bs2)

                    genbin(bs1)
                    genbin(bs2)

            genbin()
            return np.array(binary_strings, dtype=object)

        action_patterns = generate_action_patterns()
        self.rng.shuffle(action_patterns)
        train_split_idx = int(train_split * len(action_patterns))
        if self.is_train_sampler:
            return action_patterns[:train_split_idx]
        else:
            return action_patterns[train_split_idx:]

    def _get_random_trigger_action_patterns(self):
        return [#[0, 1],
                #[1, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1]
                ]

    def get_random_mixed_trigger_pattern_args(self):
        # prob = self.rng.random() * .35
        prob = 1
        mixed_strategy = [prob, 1 - prob]
        if self.rng.random() < .5:
            mixed_strategy = [1 - prob, prob]
        # triggers = np.array([[0], [1], [0, 1], [1, 0]], dtype=object)
        # trigger_actions = self.rng.choice(triggers, size=1, p=[.35, .35, .15, .15])[0]
        # if len(trigger_actions) > 1:
        #     return mixed_strategy, trigger_actions[0], 1, self.get_random_trigger_action_pattern()
        # else:
        #     return mixed_strategy, trigger_actions, self.rng.integers(1, 3), self.get_random_trigger_action_pattern()

        # trigger_actions = self.rng.choice(2)
        # return mixed_strategy, [trigger_actions], self.rng.integers(1, 3), self.get_random_trigger_action_pattern()

        # triggers = [[0, 0], [1, 1], [0, 1], [1, 0]]
        # trigger_idx = self.rng.integers(len(triggers))
        # return mixed_strategy, triggers[trigger_idx], 1, self.get_random_trigger_action_pattern()

        # triggers = [[0, 1]]
        # trigger_idx = self.rng.integers(len(triggers))
        # return mixed_strategy, triggers[trigger_idx], 1, self.get_random_trigger_action_pattern()

        trigger_actions = self.rng.choice(2)
        return mixed_strategy, [trigger_actions], 1, self.get_random_trigger_action_pattern()

    def get_random_action_pattern(self):
        pattern_idx = self.rng.integers(0, len(self.action_patterns))
        return self.action_patterns[pattern_idx]

    def get_random_trigger_action_pattern(self):
        pattern_idx = self.rng.integers(0, len(self.trigger_action_patterns))
        return self.trigger_action_patterns[pattern_idx]

    def get_random_mixed_strategy(self):
        probability_range = .9
        offset = .05
        probability = (self.rng.random() * probability_range) + offset
        return [probability, 1-probability]

    def get_random_action(self):
        action = 0
        if self.rng.random() < .5:
            action = 1
        return action

    def get_random_trigger_patience(self):
        return self.rng.integers(1, 4, endpoint=True)


def get_random_pure_strategy_agent(rss: RandomStrategySampler, agent_id: int, agent_idx: int) \
        -> Tuple[PureStrategyAgent, dict]:

    agent_action = rss.get_random_action()
    metadata = {"agent_type": "pure_strategy",
                "agent_action": agent_action}
    return PureStrategyAgent(agent_id, agent_idx, agent_action), metadata


def get_random_mixed_strategy_agent(rss: RandomStrategySampler, agent_id: int, agent_idx: int, mixed_strategy=None) \
        -> Tuple[MixedStrategyAgent, dict]:

    rand_mixed_strategy = rss.get_random_mixed_strategy()

    if mixed_strategy is None:
        mixed_strategy = rand_mixed_strategy

    metadata = {"agent_type": "mixed_strategy",
                "mixed_strategy": mixed_strategy}
    return MixedStrategyAgent(agent_id, agent_idx, mixed_strategy), metadata


def get_random_action_pattern_agent(rss: RandomStrategySampler, agent_id: int, agent_idx: int, action_pattern) \
        -> Tuple[ActionPatternAgent, dict]:

    rand_action_pattern = rss.get_random_action_pattern()

    if action_pattern is None:
        action_pattern = rand_action_pattern

    metadata = {"agent_type": "action_pattern",
                "action_pattern": action_pattern}
    return ActionPatternAgent(agent_id, agent_idx, action_pattern), metadata


def get_random_mirror_agent(rss: RandomStrategySampler, agent_id: int, agent_idx: int, starting_action=None) \
        -> Tuple[MirrorAgent, dict]:

    rand_starting_action = rss.get_random_action()

    if starting_action is None:
        starting_action = rand_starting_action

    metadata = {"agent_type": "mirror",
                "starting_action": starting_action}
    return MirrorAgent(agent_id, agent_idx, starting_action), metadata


def get_random_wsls_agent(rss: RandomStrategySampler, agent_id: int, agent_idx: int, starting_action=None,
                          win_trigger=None) -> Tuple[WSLSAgent, dict]:

    rand_starting_action = rss.get_random_action()
    rand_win_trigger = rss.get_random_action()

    if starting_action is None:
        starting_action = rand_starting_action
    if win_trigger is None:
        win_trigger = rand_win_trigger

    metadata = {"agent_type": "wsls",
                "starting_action": starting_action,
                "win_trigger": win_trigger}
    return WSLSAgent(agent_id, agent_idx, starting_action, win_trigger), metadata


def get_random_grim_trigger_agent(rss: RandomStrategySampler, agent_id: int, agent_idx: int, starting_action=None,
                                  trigger_action=None) -> Tuple[GrimTriggerAgent, dict]:

    rand_starting_action = rss.get_random_action()
    rand_trigger_action = rss.get_random_action()

    if starting_action is None:
        starting_action = rand_starting_action
    if trigger_action is None:
        trigger_action = rand_trigger_action

    metadata = {"agent_type": "grim_trigger",
                "starting_action": starting_action,
                "trigger_action": trigger_action}
    return GrimTriggerAgent(agent_id, agent_idx, starting_action, trigger_action), metadata


def get_random_trigger_pattern_agent(rss: RandomStrategySampler, agent_id: int, agent_idx: int, starting_action=None,
                                     trigger_action=None, trigger_patience=None, action_pattern=None) \
        -> Tuple[TriggerPatternAgent, dict]:

    rand_starting_action = rss.get_random_action()
    rand_trigger_action = rss.get_random_action()
    rand_trigger_patience = rss.get_random_trigger_patience()
    rand_action_pattern = rss.get_random_action_pattern()

    if starting_action is None:
        starting_action = rand_starting_action
    if trigger_action is None:
        trigger_action = rand_trigger_action
    if trigger_patience is None:
        trigger_patience = rand_trigger_patience
    if action_pattern is None:
        action_pattern = rand_action_pattern

    metadata = {"agent_type": "trigger_pattern",
                "starting_action": starting_action,
                "trigger_action": trigger_action,
                "trigger_patience": trigger_patience,
                "action_pattern": action_pattern}
    return TriggerPatternAgent(agent_id, agent_idx, starting_action, trigger_action, trigger_patience, action_pattern),\
        metadata


def get_random_action_pattern_and_mixed_strategy_agent(rss: RandomStrategySampler, agent_id: int, agent_idx: int,
                                                       action_pattern=None, mixed_strategy=None) \
        -> Tuple[ActionPatternAndMixedStrategyAgent, dict]:

    rand_action_pattern = rss.get_random_action_pattern()
    rand_mixed_strategy = rss.get_random_mixed_strategy()

    if action_pattern is None:
        action_pattern = rand_action_pattern
    if mixed_strategy is None:
        mixed_strategy = rand_mixed_strategy

    metadata = {"agent_type": "action_pattern_and_mixed_strategy",
                "action_pattern": action_pattern,
                "mixed_strategy": mixed_strategy}
    return ActionPatternAndMixedStrategyAgent(agent_id, agent_idx, action_pattern, mixed_strategy), metadata


def get_random_action_pattern_and_mirror_agent(rss: RandomStrategySampler, agent_id: int, agent_idx: int,
                                               action_pattern=None) -> Tuple[ActionPatternAndMirrorAgent, dict]:

    rand_action_pattern = rss.get_random_action_pattern()

    if action_pattern is None:
        action_pattern = rand_action_pattern

    metadata = {"agent_type": "action_pattern_and_mirror",
                "action_pattern": action_pattern}
    return ActionPatternAndMirrorAgent(agent_id, agent_idx, action_pattern), metadata


def get_random_mixed_trigger_pattern_agent(rss: RandomStrategySampler, agent_id: int, agent_idx: int,
                                           mixed_strategy=None, trigger_action=None, trigger_patience=None,
                                           action_pattern=None) -> Tuple[MixedTriggerPattern, dict]:

    rand_mixed_strategy, rand_trigger_action, rand_trigger_patience, rand_action_pattern = rss.get_random_mixed_trigger_pattern_args()

    if mixed_strategy is None:
        mixed_strategy = rand_mixed_strategy
    if trigger_action is None:
        trigger_action = rand_trigger_action
    if trigger_patience is None:
        trigger_patience = rand_trigger_patience
    if action_pattern is None:
        action_pattern = rand_action_pattern

    metadata = {"agent_type": "mixed_trigger_pattern",
                "mixed_strategy": mixed_strategy,
                "trigger_action": trigger_action,
                "trigger_patience": trigger_patience,
                "action_pattern": action_pattern}
    return MixedTriggerPattern(agent_id, agent_idx, mixed_strategy, trigger_action, trigger_patience=trigger_patience,
                               action_pattern=action_pattern), metadata


def get_random_iterative_action_agent(agent_type: str, rss: RandomStrategySampler, agent_id: int, agent_idx: int,
                                      **kwargs) -> Tuple[IterativeActionAgent, dict]:

    get_random_agent_funcs = {
        "pure_strategy": get_random_pure_strategy_agent,
        "mixed_strategy": get_random_mixed_strategy_agent,
        "action_pattern": get_random_action_pattern_agent,
        "mirror": get_random_mirror_agent,
        "wsls": get_random_wsls_agent,
        "grim_trigger": get_random_grim_trigger_agent,
        "trigger_pattern": get_random_trigger_pattern_agent,
        "action_pattern_and_mixed_strategy": get_random_action_pattern_and_mixed_strategy_agent,
        "action_pattern_and_mirror": get_random_action_pattern_and_mirror_agent,
        "mixed_trigger_pattern": get_random_mixed_trigger_pattern_agent,
    }

    if agent_type not in get_random_agent_funcs:
        raise ValueError(f"agent_type {agent_type} is not recognized. Please choose from the following agent type's "
                         f"{list(get_random_agent_funcs.keys())}")
    return get_random_agent_funcs[agent_type](rss, agent_id, agent_idx, **kwargs)
