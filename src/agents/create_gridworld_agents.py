import itertools

import numpy as np

from src.agents.gridworld_agents import PathfindingAgent
from src.agents.destination_selector import DestinationSelector, MultiAgentDestinationSelector, \
    DiscountGoalRanker, ClosestDistanceGoalRanker, HighestGoalRanker

from src.agents.reward_function import get_cooperative_reward_functions

from typing import Tuple, List


class RandomAgentParamSampler:
    def __init__(self, seed: int, is_train_sampler: bool):
        if not is_train_sampler:
            seed += 1
        self.rng = np.random.default_rng(seed)
        self.goal_ranker_types = ["highest", "discount", "closest"]
        self.discounts = [.75]
        self.goal_rewards = []
        num_goals_per_group = 4
        for goal_value_permutation in itertools.permutations([1, .5, .1]):
            goal_rewards = []
            for goal_value in goal_value_permutation:
                goal_rewards += [goal_value for _ in range(num_goals_per_group)]
            self.goal_rewards.append(goal_rewards)

    def get_random_filter(self):
        return "none"
        # if self.rng.random() < self.prob_none_filter:
        #     return "none"
        # else:
        #     return self.filters[self.rng.choice(len(self.filters))]

    def get_filter(self, filter_type, **kwargs):
        return None
        # if filter_type == "none":
        #     return None
        # elif filter_type == "square":
        #     return StaticSquareFilter(**kwargs)
        # elif filter_type == "diamond":
        #     return StaticDiamondFilter(**kwargs)
        # elif filter_type == "state_partition":
        #     return CollaborativeStatePartitionFilter(**kwargs, seed=self.rng.integers(1, 100000000))

    def get_random_goal_ranker_type(self):
        return self.rng.choice(self.goal_ranker_types)

    def get_goal_ranker(self, goal_ranker_type, collaborative: bool, discount=None):
        if goal_ranker_type == "highest":
            return HighestGoalRanker(collaborative)
        elif goal_ranker_type == "discount":
            return DiscountGoalRanker(self.discounts[0])
        elif goal_ranker_type == "closest":
            return ClosestDistanceGoalRanker()
        else:
            raise ValueError(f"Don't recognize goal_ranker_type={goal_ranker_type}")

    def get_random_goal_rewards(self, num_goals=12):
        return self.goal_rewards[self.rng.choice(len(self.goal_rewards))]


def get_random_collaborative_agents(raps: RandomAgentParamSampler, agent_ids, goal_rewards=None, num_goals=12,
                                    goal_ranker_type=None, discount=None, ):
    num_agents = len(agent_ids)

    if goal_rewards is None:
        goal_rewards = raps.get_random_goal_rewards(num_goals)
    if goal_ranker_type is None:
        goal_ranker_type = raps.get_random_goal_ranker_type()

    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards)
    goal_ranker = raps.get_goal_ranker(goal_ranker_type, collaborative=True, discount=discount)
    metadata = {"agent_type": "collaborative", "goal_rewards": goal_rewards, "goal_ranker": goal_ranker_type}
    if isinstance(goal_ranker, DiscountGoalRanker):
        metadata["discount"] = goal_ranker.discount
    agents = []
    destination_selectors = [MultiAgentDestinationSelector(reward_funcs[0], 0, goal_ranker, True)]
    for agent_idx in range(1, num_agents):
        destination_selector = MultiAgentDestinationSelector(reward_funcs[agent_idx], agent_idx, goal_ranker, False)
        destination_selectors[0].add_destination_selector(destination_selector)
        destination_selectors.append(destination_selector)

    for agent_idx in range(num_agents):
        agents.append(PathfindingAgent(agent_ids[agent_idx], reward_funcs[agent_idx], destination_selectors[agent_idx]))

    return agents, [metadata.copy() for _ in range(num_agents)]


def get_random_independent_agents(raps: RandomAgentParamSampler, agent_ids, goal_rewards=None, num_goals=12,
                                  goal_ranker_type=None, discount=None):
    num_agents = len(agent_ids)

    if goal_rewards is None:
        goal_rewards = raps.get_random_goal_rewards(num_goals)

    if goal_ranker_type is None:
        goal_ranker_type = raps.get_random_goal_ranker_type()

    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards)
    metadata = []
    agents = []
    for agent_idx in range(num_agents):
        goal_ranker = raps.get_goal_ranker(goal_ranker_type, collaborative=False, discount=discount)
        metadata.append({"agent_type": "independent", "goal_rewards": goal_rewards,
                         "goal_ranker": goal_ranker_type})
        if isinstance(goal_ranker, DiscountGoalRanker):
            metadata[-1]["discount"] = goal_ranker.discount
        destination_selector = DestinationSelector(reward_funcs[agent_idx], agent_idx, goal_ranker, None)
        agents.append(PathfindingAgent(agent_ids[agent_idx], reward_funcs[agent_idx], destination_selector))
    return agents, metadata


def get_random_gridworld_agent(agent_type: str, raps: RandomAgentParamSampler, agent_ids, **agent_kwargs) \
        -> Tuple[List[PathfindingAgent], List[dict]]:
    if agent_type == "independent":
        return get_random_independent_agents(raps, agent_ids, **agent_kwargs)
    elif agent_type == "collaborative":
        return get_random_collaborative_agents(raps, agent_ids, **agent_kwargs)
    else:
        raise ValueError(f"Don't recognize agent_type {agent_type}")
