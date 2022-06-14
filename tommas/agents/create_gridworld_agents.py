import itertools

import numpy as np

from tommas.agents.hand_coded_agents import PathfindingAgent
from tommas.agents.destination_selector import DestinationSelector, MultiAgentDestinationSelector, \
    DiscountGoalRanker, ClosestDistanceGoalRanker, HighestGoalRanker, StaticDiamondFilter, StaticSquareFilter, \
    CollaborativeStatePartitionFilter

from tommas.agents.reward_function import get_cooperative_reward_functions

from typing import Tuple, List


class RandomAgentParamSampler:
    def __init__(self, seed: int, is_train_sampler: bool):
        if not is_train_sampler:
            seed += 1
        self.rng = np.random.default_rng(seed)
        # self.prob_none_filter = .2
        # self.filters = []
        # for radius in [2, 4, 6]:
        #     self.filters.append(("square", radius))
        # for radius in [4, 6, 8]:
        #     self.filters.append(("diamond", radius))
        # self.goal_ranker_types = ["highest", "closest", "discount"]
        self.goal_ranker_types = ["highest", "discount", "closest"]
        # self.discounts = [.5, .9, .99]
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
        # if goal_ranker_type == "highest":
        #     return HighestGoalRanker(collaborative)
        # elif goal_ranker_type == "closest":
        #     return ClosestDistanceGoalRanker()
        # else:
        #     if discount is None:
        #         discount = self.rng.choice(self.discounts)
        #     return DiscountGoalRanker(discount)

        if goal_ranker_type == "highest":
            return HighestGoalRanker(collaborative)
        elif goal_ranker_type == "discount":
            return DiscountGoalRanker(self.discounts[0])
        elif goal_ranker_type == "closest":
            return ClosestDistanceGoalRanker()
        else:
            raise ValueError(f"Don't recognize goal_ranker_type={goal_ranker_type}")

    def get_random_goal_rewards(self, num_goals=12):
        # goal_rewards = self.rng.dirichlet(alpha=np.ones(num_goals))
        # num_negative_goals = self.rng.choice(4, size=None, p=[.4, .3, .2, .1]) * 3 + self.rng.integers(0, 3)
        # goal_rewards[np.argsort(goal_rewards)[:num_negative_goals]] *= -1
        # # goal_rewards -= self.rng.random() / num_goals  # Another potential way to have random negative rewards
        # return goal_rewards * 2  # Multiply by 2 so chance of having goals == 1. This helps to destination selector parameters such as distance_penalty
        return self.goal_rewards[self.rng.choice(len(self.goal_rewards))]


# def get_random_state_partition_agents(raps: RandomAgentParamSampler, agent_ids, goal_rewards=None, num_goals=12,
#                                       goal_ranker_type=None, discount=None, ):
#     num_agents = len(agent_ids)
#
#     if goal_rewards is None:
#         goal_rewards = raps.get_random_goal_rewards(num_goals)
#     if goal_ranker_type is None:
#         goal_ranker_type = raps.get_random_goal_ranker_type()
#
#     reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards)
#     goal_ranker = raps.get_goal_ranker(goal_ranker_type, collaborative=False, discount=discount)
#     state_filter = raps.get_filter("state_partition", agent_indices=list(range(num_agents)),
#                                    agent_idx_offset=1+num_goals)
#     metadata = {"agent_type": "state_partition", "goal_rewards": goal_rewards, "goal_ranker": goal_ranker_type,
#                 "filter": "state_partition"}
#     if isinstance(goal_ranker, DiscountGoalRanker):
#         metadata["discount"] = goal_ranker.discount
#     agents = []
#     destination_selectors = [MultiAgentDestinationSelector(reward_funcs[0], 0, goal_ranker, True, state_filter)]
#     for agent_idx in range(1, num_agents):
#         destination_selector = MultiAgentDestinationSelector(reward_funcs[agent_idx], agent_idx, goal_ranker, False,
#                                                              state_filter)
#         destination_selectors[0].add_destination_selector(destination_selector)
#         destination_selectors.append(destination_selector)
#
#     for agent_idx in range(num_agents):
#         agents.append(PathfindingAgent(agent_ids[agent_idx], reward_funcs[agent_idx], destination_selectors[agent_idx]))
#
#     return agents, [metadata.copy() for _ in range(num_agents)]


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
# def get_random_collaborative_agents(raps: RandomAgentParamSampler, agent_ids, goal_rewards=None, num_goals=12,
#                                          goal_ranker_type=None, discount=None, ):
#     num_agents = len(agent_ids)
#
#     if goal_rewards is None:
#         goal_rewards = raps.get_random_goal_rewards(num_goals)
#     if goal_ranker_type is None:
#         goal_ranker_type = raps.get_random_goal_ranker_type()
#
#     reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards)
#     goal_ranker = raps.get_goal_ranker(goal_ranker_type, collaborative=True, discount=discount)
#     metadata = {"agent_type": "collaborative", "goal_rewards": goal_rewards, "filter": "none",
#                 "goal_ranker": goal_ranker_type}
#     if isinstance(goal_ranker, DiscountGoalRanker):
#         metadata["discount"] = goal_ranker.discount
#     agents = []
#     destination_selectors = [MultiAgentDestinationSelector(reward_funcs[0], 0, goal_ranker, True)]
#     for agent_idx in range(1, num_agents):
#         destination_selector = MultiAgentDestinationSelector(reward_funcs[agent_idx], agent_idx, goal_ranker, False)
#         destination_selectors[0].add_destination_selector(destination_selector)
#         destination_selectors.append(destination_selector)
#
#     for agent_idx in range(num_agents):
#         agents.append(PathfindingAgent(agent_ids[agent_idx], reward_funcs[agent_idx], destination_selectors[agent_idx]))
#
#     return agents, [metadata.copy() for _ in range(num_agents)]


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
# def get_random_independent_agents(raps: RandomAgentParamSampler, agent_ids, goal_rewards=None, num_goals=12,
#                                   goal_ranker_type=None, discount=None, state_filter_type=None,
#                                   single_modelled_agent=False):
#     num_agents = len(agent_ids)
#
#     if goal_rewards is None:
#         goal_rewards = raps.get_random_goal_rewards(num_goals)
#     if single_modelled_agent:
#         goal_ranker_type = [goal_ranker_type] + [raps.get_random_goal_ranker_type() for _ in range(num_agents - 1)]
#         state_filter_type = [state_filter_type] + [raps.get_random_filter() for _ in range(num_agents - 1)]
#     else:
#         if goal_ranker_type is None:
#             goal_ranker_type = [raps.get_random_goal_ranker_type() for _ in range(num_agents)]
#         if state_filter_type is None:
#             state_filter_type = [raps.get_random_filter() for _ in range(num_agents)]
#
#     reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards)
#     metadata = []
#     agents = []
#     for agent_idx in range(num_agents):
#         goal_ranker = raps.get_goal_ranker(goal_ranker_type, collaborative=False, discount=discount)
#         agent_filter_type = state_filter_type[agent_idx]
#         if isinstance(agent_filter_type, tuple):# or isinstance(agent_filter_type, list) or isinstance(agent_filter_type, list):
#             state_filter = raps.get_filter(agent_filter_type[0], radius=agent_filter_type[1])
#             agent_filter_type = f"{agent_filter_type[0]}_{agent_filter_type[1]}"
#         else:
#             state_filter = raps.get_filter(agent_filter_type)
#         metadata.append({"agent_type": "independent", "goal_rewards": goal_rewards,
#                          "filter": agent_filter_type, "goal_ranker": goal_ranker_type[agent_idx]})
#         if isinstance(goal_ranker, DiscountGoalRanker):
#             metadata[-1]["discount"] = goal_ranker.discount
#         destination_selector = DestinationSelector(reward_funcs[agent_idx], agent_idx, goal_ranker, state_filter)
#         agents.append(PathfindingAgent(agent_ids[agent_idx], reward_funcs[agent_idx], destination_selector))
#     return agents, metadata


def get_random_gridworld_agent(agent_type: str, raps: RandomAgentParamSampler, agent_ids, **agent_kwargs) \
        -> Tuple[List[PathfindingAgent], List[dict]]:
    if agent_type == "independent":
        return get_random_independent_agents(raps, agent_ids, **agent_kwargs)
    # elif agent_type == "state_partition":
    #     return get_random_state_partition_agents(raps, agent_ids, **agent_kwargs)
    elif agent_type == "collaborative":
        return get_random_collaborative_agents(raps, agent_ids, **agent_kwargs)
    else:
        raise ValueError(f"Don't recognize agent_type {agent_type}")
