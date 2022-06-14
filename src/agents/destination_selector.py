import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from typing import Union, List

from src.agents.reward_function import RewardFunction
from src.helper_code.navigation import astar


class GoalRanker:
    def rank(self, state, agent_idx, agent_pos, goal_rewards, reachable_goals):
        return np.ones(len(goal_rewards[0])) * -np.inf


class HighestGoalRanker(GoalRanker):
    def __init__(self, collaborative):
        self.collaborative = collaborative

    def rank(self, state, agent_idx, agent_pos, goal_rewards, reachable_goals):
        goal_ranking = super().rank(state, agent_idx, agent_pos, goal_rewards, reachable_goals)
        max_distance = 1000
        if self.collaborative:
            goal_ranking = list(goal_ranking)
            distance_bonus = np.zeros_like(goal_ranking)
            for goal in range(len(goal_ranking)):
                goal_value = goal_rewards[agent_idx][goal]
                if goal_value > 0 and (goal in reachable_goals):
                    goal_position = reachable_goals[goal]
                    path = astar(state[0], agent_pos, goal_position)
                    distance_bonus[goal] = 1 - (len(path) / max_distance)
                    goal_ranking[goal] = (goal_value, goal)
                else:
                    goal_ranking[goal] = (-np.inf, goal)
            sorted_goal_ranking = sorted(goal_ranking, key=lambda x: x[0])
            for new_goal_value in range(len(goal_ranking)):
                current_goal_value, goal = sorted_goal_ranking[new_goal_value]
                if current_goal_value != -np.inf:
                    goal_ranking[goal] = new_goal_value
                else:
                    goal_ranking[goal] = current_goal_value
            goal_ranking = goal_ranking + distance_bonus
        else:
            for goal in reachable_goals:
                goal_value = goal_rewards[agent_idx][goal]
                if goal_value > 0:
                    goal_ranking[goal] = goal_value
        return goal_ranking


class DistanceGoalRanker(GoalRanker):
    def rank(self, state, agent_idx, agent_pos, goal_rewards, reachable_goals):
        goal_ranking = super().rank(state, agent_idx, agent_pos, goal_rewards, reachable_goals)
        if reachable_goals:
            for goal, goal_position in reachable_goals.items():
                if goal_rewards[agent_idx][goal] > 0:
                    path = astar(state[0], agent_pos, goal_position)
                    distance_to_goal = len(path) - 1  # Subtract one because current position in path
                    greedy_value = self.greedy_metric(goal_rewards[agent_idx][goal], distance_to_goal)
                    goal_ranking[goal] = greedy_value
        return goal_ranking

    def greedy_metric(self, goal_value, distance_to_goal):
        raise NotImplementedError


class ClosestDistanceGoalRanker(DistanceGoalRanker):
    def greedy_metric(self, goal_value, distance_to_goal):
        # adding (goal_value * .0001) so high value goal with same distance as low value gets preferred
        return -distance_to_goal + (goal_value * .0001)


class DiscountGoalRanker(DistanceGoalRanker):
    def __init__(self, discount):
        self.discount = discount

    def greedy_metric(self, goal_value, distance_to_goal):
        return goal_value * (self.discount ** distance_to_goal)


class DistancePenaltyGoalRanker(DistanceGoalRanker):
    def __init__(self, distance_penalty):
        self.distance_penalty = distance_penalty

    def greedy_metric(self, goal_value, distance_to_goal):
        return goal_value + (self.distance_penalty * distance_to_goal)


class StateFilter:
    def __init__(self, iterative_update):
        self.iterative_update = iterative_update

    def reset(self):
        raise NotImplementedError

    def filter(self, state, agent_position) -> set:
        raise NotImplementedError


class StaticShapeFilter(StateFilter):
    def __init__(self, radius):
        super().__init__(iterative_update=False)
        if radius <= 0:
            raise ValueError("radius should be > 0")
        self.radius = radius
        self.state_filter = None

    def reset(self):
        self.state_filter = None

    def _assign_state_filter(self, state, agent_position):
        raise NotImplementedError

    def filter(self, state, agent_position):
        if self.state_filter is None:
            self._assign_state_filter(state, agent_position)
        return self.state_filter


class StaticSquareFilter(StaticShapeFilter):
    def _assign_state_filter(self, state, agent_position):
        positions = []
        for i in range(agent_position[0] - self.radius, agent_position[0] + self.radius + 1):
            for j in range(agent_position[1] - self.radius, agent_position[1] + self.radius + 1):
                positions.append((i, j))
        self.state_filter = set(positions)


class StaticDiamondFilter(StaticShapeFilter):
    def _assign_state_filter(self, state, agent_position):
        positions = []
        for i in range(agent_position[0] - self.radius, agent_position[0] + self.radius + 1):
            offset = self.radius - abs(agent_position[0] - i)
            for j in range(agent_position[1] - offset, agent_position[0] + offset + 1):
                positions.append((i, j))
        self.state_filter = set(positions)


class StaticCircleFilter(StaticShapeFilter):
    def _assign_state_filter(self, state, agent_position):
        def tuple_distance(tupleA, tupleB):
            return np.linalg.norm((tupleA[0] - tupleB[0], tupleA[1] - tupleB[1]))

        positions = []
        for i in range(agent_position[0] - self.radius, agent_position[0] + self.radius + 1):
            for j in range(agent_position[1] - self.radius, agent_position[1] + self.radius + 1):
                distance = tuple_distance(agent_position, (i, j))
                if distance <= self.radius:
                    positions.append((i, j))
        self.state_filter = set(positions)


class DestinationSelector:
    def __init__(self,
                 reward_func: RewardFunction,
                 agent_idx: int,
                 goal_ranker: GoalRanker,
                 state_filters: Union[List[StateFilter], StateFilter, None] = None,
                 ):
        self.agent_idx = agent_idx
        self.reward_func = reward_func
        self.update_state_filter = False
        self._init_state_filters(state_filters)
        self.goal_rewards = self.reward_func.goal_rewards
        self.num_goals = self.goal_rewards.shape[1]
        self.reachable_positions = None
        self.all_reachable_goals = {}
        self.current_reachable_goals = {}
        self.goal_ranker = goal_ranker
        self.current_goal = None
        self.current_destination = None

    def _init_state_filters(self, state_filters):
        if isinstance(state_filters, StateFilter):
            self.state_filters = [state_filters]
        else:
            self.state_filters = state_filters
        iterative_update_idx = -1
        if self.state_filters is not None:
            for i, state_filter in enumerate(self.state_filters):
                if state_filter.iterative_update is True:
                    iterative_update_idx = i
                    break
        if iterative_update_idx >= 0:
            self.state_filters[0], self.state_filters[iterative_update_idx] = \
                self.state_filters[iterative_update_idx], self.state_filters[0]

    def _get_agent_current_position(self, state):
        agent_channel = state[self.num_goals + 1 + self.agent_idx]
        return list(zip(*np.where(agent_channel == 1)))[0]

    def reset(self, reachable_positions: list, state: np.array):
        if self.state_filters is not None:
            for state_filter in self.state_filters:
                state_filter.reset()
        self.remove_destination()
        self.all_reachable_goals = {}
        self.current_reachable_goals = {}
        self.reachable_positions = reachable_positions
        self.get_reachable_goals(state)

    def _state_filter_update_required(self):
        if self.state_filters is None:
            return False
        return self.state_filters[0].iterative_update

    def _update_current_reachable_goals(self, state):
        if self.state_filters is None:
            self.current_reachable_goals = self.all_reachable_goals.copy()
            return

        agent_position = self._get_agent_current_position(state)
        filtered_positions = self.state_filters[0].filter(state, agent_position)
        for i in range(1, len(self.state_filters)):
            filtered_positions = filtered_positions.intersection(self.state_filters[i].filter(state, agent_position))

        self.current_reachable_goals = dict()
        for goal, goal_position in self.all_reachable_goals.items():
            if goal_position in filtered_positions:
                self.current_reachable_goals[goal] = goal_position

    def get_reachable_goals(self, state):
        for position in self.reachable_positions:
            for goal in range(self.num_goals):
                if state[(goal + 1, *position)] == 1:
                    self.all_reachable_goals[goal] = (position[0], position[1])
                    self.current_reachable_goals[goal] = (position[0], position[1])
        self._update_current_reachable_goals(state)

    def set_goal_destination(self, goal):
        self.current_destination = self.current_reachable_goals[goal]
        self.current_goal = goal

    def set_destination(self, destination):
        self.current_destination = destination
        self.current_goal = None

    def remove_reached_goals(self, state):
        goals_to_remove = []
        for goal, goal_position in self.all_reachable_goals.items():
            if state[(1 + goal, *goal_position)] == 0:
                goals_to_remove.append(goal)
        for goal in goals_to_remove:
            del self.all_reachable_goals[goal]
            if goal in self.current_reachable_goals:
                del self.current_reachable_goals[goal]

    def remove_destination(self):
        self.current_goal = None
        self.current_destination = None

    def destination_reached(self, state):
        if self.current_goal is not None:
            if state[(1 + self.current_goal, *self.current_destination)] == 0:
                self.remove_destination()
                return True
        if self.current_destination is not None:
            agent_pos = self._get_agent_current_position(state)
            if agent_pos == self.current_destination:
                self.remove_destination()
                return True
        return False

    def assign_destination(self, state):
        if self.current_reachable_goals:
            goal_rankings = self.goal_ranker.rank(state, self.agent_idx, self._get_agent_current_position(state),
                                                  self.goal_rewards, self.current_reachable_goals)
            max_goal = np.argmax(goal_rankings)
            if goal_rankings[max_goal] > -np.inf:
                self.set_goal_destination(max_goal)
                return
        self.set_destination(None)

    def get_destination(self, state):
        self.remove_reached_goals(state)
        if self._state_filter_update_required():
            self._update_current_reachable_goals(state)
        if self.destination_reached(state) or self.current_destination is None:
            self.assign_destination(state)
        return self.current_destination


class MultiAgentDestinationSelector(DestinationSelector):
    def __init__(self, reward_func, agent_idx, goal_ranker, is_primary_selector, state_filters=None):
        super().__init__(reward_func, agent_idx, goal_ranker, state_filters)
        self.is_primary_selector = is_primary_selector
        self.destination_selectors = [self]
        self.num_destination_selectors = 1

    def add_destination_selector(self, destination_selector):
        if self.is_primary_selector is False:
            raise ValueError("You shouldn't be calling add_destination_selector on a non-primary destination selector")
        if destination_selector.is_primary_selector is True:
            raise ValueError("You shouldn't be adding a primary destination selector to another selector")
        self.destination_selectors.append(destination_selector)
        self.num_destination_selectors += 1

    def assign_destination(self, state):
        combined_goal_rankings = []
        for destination_selector in self.destination_selectors:
            goal_rankings = destination_selector.goal_ranker.rank(state, destination_selector.agent_idx,
                                                                  destination_selector._get_agent_current_position(state),
                                                                  destination_selector.goal_rewards,
                                                                  destination_selector.current_reachable_goals)
            combined_goal_rankings.append(goal_rankings)
        combined_goal_rankings = np.array(combined_goal_rankings)
        invalid_indices = combined_goal_rankings == -np.inf
        combined_goal_rankings[invalid_indices] = np.inf
        combined_goal_rankings += abs(np.min(combined_goal_rankings)) + 1
        combined_goal_rankings[invalid_indices] = 0
        partition = [i for i in zip(*linear_sum_assignment(combined_goal_rankings, maximize=True))]
        remove_indices = []
        for i in partition:
            if combined_goal_rankings[i] == 0:  # 0 is unreachable goal
                remove_indices.append(i)
        for i in remove_indices:
            partition.remove(i)
        for selector_idx, goal in partition:
            self.destination_selectors[selector_idx].set_goal_destination(goal)

    def get_destination(self, state):
        if self.is_primary_selector:
            for destination_selector in self.destination_selectors:
                destination_selector.remove_reached_goals(state)
                if destination_selector._state_filter_update_required():
                    destination_selector._update_current_reachable_goals(state)

            for destination_selector in self.destination_selectors:
                if destination_selector.destination_reached(state) or destination_selector.current_destination is None:
                    self.assign_destination(state)
                    break
        return self.current_destination
