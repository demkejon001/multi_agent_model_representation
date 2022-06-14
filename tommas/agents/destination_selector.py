import itertools

import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from typing import Optional, Union, List

from tommas.agents.reward_function import RewardFunction
from tommas.helper_code.navigation import maze_traversal, astar


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


class CollaborativeStatePartitionFilter(StateFilter):
    def __init__(self, agent_indices, agent_idx_offset, seed=None):
        super().__init__(iterative_update=False)
        self.agent_indices = agent_indices
        self.agent_idx_offset = agent_idx_offset
        self.partitions = None
        if seed is None:
            seed = np.random.randint(0, 10000000)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def reset(self):
        self.partitions = None
        self.rng = np.random.default_rng(self.seed)

    def _get_agent_partition(self, state, agent_position):
        for partition_idx, agent_idx in enumerate(self.agent_indices):
            if state[(self.agent_idx_offset + agent_idx, *agent_position)] == 1:  # If agent lies in agent_position
                return self.partitions[partition_idx]
        raise ValueError("agent_position does not correlate with any of the initial agent_indices provided")

    def filter(self, state, agent_position) -> set:
        if self.partitions is None:
            self.partitions = self._get_state_partition(state)
        return set(self._get_agent_partition(state, agent_position))

    @staticmethod
    def _find_closest_integer_factor_pair(n):
        potential_factor = int(math.sqrt(n))
        while n % potential_factor != 0:
            potential_factor -= 1
        return potential_factor, n // potential_factor

    # rows <= cols and if num_destination_selectors is prime, then rows is 1
    def _get_state_partition_row_cols(self):
        combine_room = False
        rows, cols = self._find_closest_integer_factor_pair(len(self.agent_indices))
        if rows == 1 and len(self.agent_indices) != 2:
            combine_room = True
            rows, cols = self._find_closest_integer_factor_pair(len(self.agent_indices) + 1)
        return rows, cols, combine_room

    @staticmethod
    def _get_unbordered_state_indices(state):
        _, state_rows, state_cols = state.shape
        unbordered_state_indices = np.arange(state_rows*state_cols).reshape((state_rows, state_cols))
        unbordered_state_indices = np.delete(unbordered_state_indices, [0, -1], axis=0)
        unbordered_state_indices = np.delete(unbordered_state_indices, [0, -1], axis=1)
        return unbordered_state_indices

    def _get_random_adjacent_rooms(self, rows, cols):
        combine_row_adjacent_room = self.rng.random() < .5
        if combine_row_adjacent_room:
            random_row = self.rng.integers(0, rows - 1)
            random_col = self.rng.integers(0, cols)
            adjacent_row = random_row + 1
            adjacent_col = random_col
        else:  # combine column adjacent room
            random_row = self.rng.integers(0, rows)
            random_col = self.rng.integers(0, cols - 1)
            adjacent_row = random_row
            adjacent_col = random_col + 1
        return (random_row, random_col), (adjacent_row, adjacent_col), combine_row_adjacent_room

    def _combine_room(self, partitions, rows, cols):
        room1, room2, combine_row_adjacent_room = self._get_random_adjacent_rooms(rows, cols)
        room1_idx = np.ravel_multi_index(room1, (rows, cols))
        room2_idx = np.ravel_multi_index(room2, (rows, cols))
        partitions[room1_idx] += partitions[room2_idx]
        partitions.pop(room2_idx)

    def _partition_unbordered_state_indices(self, unbordered_state_indices, rows, cols, combine_room):
        def array_hsplit(arr, num_splits):
            splits = []
            split_indices = np.array_split(np.arange(arr.shape[1]), num_splits)
            for split_idx in split_indices:
                splits.append(arr[:, split_idx])
            return splits

        def array_vsplit(arr, num_splits):
            splits = []
            split_indices = np.array_split(np.arange(arr.shape[0]), num_splits)
            for split_idx in split_indices:
                splits.append(arr[split_idx, :])
            return splits

        partitions = []
        unbordered_state_indices_vsplits = array_hsplit(unbordered_state_indices, cols)
        for unbordered_state_indices_vsplit in unbordered_state_indices_vsplits:
             partitions += [arr.flatten().tolist() for arr in array_vsplit(unbordered_state_indices_vsplit, rows)]
        if combine_room:
            self._combine_room(partitions, rows, cols)
        return partitions

    def _get_state_partition(self, state):
        unbordered_state_indices = self._get_unbordered_state_indices(state)
        rows, cols, combine_room = self._get_state_partition_row_cols()
        partitions = self._partition_unbordered_state_indices(unbordered_state_indices, rows, cols, combine_room)
        for i in range(len(partitions)):
            partitions[i] = [pos for pos in zip(*np.unravel_index(partitions[i], (state.shape[1], state.shape[2])))]
        return partitions


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


class PicnicDestinationSelector(MultiAgentDestinationSelector):
    def __init__(self, reward_func, agent_idx, goal_ranker, is_primary_selector, local_state_filter=None):
        super().__init__(reward_func, agent_idx, goal_ranker, is_primary_selector, local_state_filter)
        self.find_picnic_food = False
        self.picnic_pairs = []
        self.no_more_picnics = False
        self.call_init_picnic_pairs_required = True
        self.picnic_pair_ctr = -1
        self.destinations_reached = []

    def reset(self, reachable_positions, state):
        super(PicnicDestinationSelector, self).reset(reachable_positions, state)
        if self.is_primary_selector:
            self.find_picnic_food = False
            self.picnic_pairs = []
            self.no_more_picnics = False
            self.call_init_picnic_pairs_required = True
            self.picnic_pair_ctr = -1
            self.destinations_reached = [True for _ in range(self.num_destination_selectors)]

    def _init_picnic_pairs(self):
        self.picnic_pairs = [[]]
        if self.num_destination_selectors == 2:
            if self.destination_selectors[0].reachable_positions[0] in self.destination_selectors[1].reachable_positions:
                self.picnic_pairs = [[[0, 1]]]
        elif self.num_destination_selectors == 4:
            reachable_goals_to_agents = dict()
            for i, dest_selector in enumerate(self.destination_selectors):
                reachable_goals = tuple(np.sort(tuple(dest_selector.all_reachable_goals.keys())))
                if len(reachable_goals) == 0:
                    continue
                if reachable_goals not in reachable_goals_to_agents:
                    reachable_goals_to_agents[reachable_goals] = []
                reachable_goals_to_agents[reachable_goals].append(i)
            reachable_agent_groups = [reachable_agents for reachable_agents in reachable_goals_to_agents.values() if len(reachable_agents) > 1]
            if len(reachable_agent_groups) == 0:
                self.picnic_pairs = [[]]
            elif len(reachable_agent_groups) == 2:
                self.picnic_pairs = [reachable_agent_groups.copy()]
            else:  # There is 1 group
                reachable_agents = reachable_agent_groups[0]
                if len(reachable_agents) == 2:
                    self.picnic_pairs = [[reachable_agents.copy()]]
                elif len(reachable_agents) == 3:
                    self.picnic_pairs = []
                    for pair in itertools.combinations(reachable_agents, 2):
                        self.picnic_pairs.append([list(pair)])
                else:
                    self.picnic_pairs = [[[0, 1], [2, 3]],
                                         [[0, 2], [1, 3]],
                                         [[0, 3], [1, 2]]]
        else:
            raise ValueError("At the moment PicnicCommunicator can only handle 2 or 4 destination selectors")

    def _assign_picnic_pair_positions(self, state):
        self.destinations_reached = [True for _ in range(self.num_destination_selectors)]
        current_picnic_pairs = self.picnic_pairs[self.picnic_pair_ctr]
        for agent_pair in current_picnic_pairs:
            a_dest_selector = self.destination_selectors[agent_pair[0]]
            b_dest_selector = self.destination_selectors[agent_pair[1]]
            a_position = a_dest_selector._get_agent_current_position(state)
            b_position = b_dest_selector._get_agent_current_position(state)
            path = astar(state[0], a_position, b_position)
            a_meetup_point_idx = (len(path)-1) // 2
            a_dest_selector.set_destination(path[a_meetup_point_idx])
            b_dest_selector.set_destination(path[a_meetup_point_idx + 1])
            self.destinations_reached[agent_pair[0]] = False
            self.destinations_reached[agent_pair[1]] = False

    def all_destinations_reached(self, state):
        for i, dest_selector in enumerate(self.destination_selectors):
            if dest_selector.current_destination is None:
                self.destinations_reached[i] = True
            if not self.destinations_reached[i]:
                self.destinations_reached[i] = dest_selector.destination_reached(state)
        destination_reached = all(self.destinations_reached)
        if destination_reached:
            self.destinations_reached = [False for _ in range(len(self.destination_selectors))]
        return destination_reached

    def _any_reachable_goals_in_destination_selectors(self):
        for destination_selector in self.destination_selectors:
            if destination_selector.current_reachable_goals:
                return True
        return False

    def assign_destination(self, state):
        if self.find_picnic_food:
            super().assign_destination(state)
            if not self._any_reachable_goals_in_destination_selectors():
                self.no_more_picnics = True
        else:
            if self.no_more_picnics:
                return
            self._assign_picnic_pair_positions(state)

    def _update_picnic_pair_ctr(self):
        if len(self.picnic_pairs) == 0:
            self.picnic_pair_ctr = 0
        else:
            self.picnic_pair_ctr = (self.picnic_pair_ctr + 1) % len(self.picnic_pairs)

    def get_destination(self, state):
        if self.is_primary_selector:
            if self.call_init_picnic_pairs_required:
                self.call_init_picnic_pairs_required = False
                self._init_picnic_pairs()
            for destination_selector in self.destination_selectors:
                destination_selector.remove_reached_goals(state)
                if destination_selector._state_filter_update_required():
                    destination_selector._update_current_reachable_goals(state)

            if self.all_destinations_reached(state):
                if self.find_picnic_food:
                    self._update_picnic_pair_ctr()
                self.find_picnic_food = not self.find_picnic_food
                self.assign_destination(state)
        return self.current_destination


class SinglePicnicDestinationSelector(PicnicDestinationSelector):
    def __init__(self, reward_func, agent_idx, goal_ranker, is_primary_selector, local_state_filter=None):
        super().__init__(reward_func, agent_idx, goal_ranker, is_primary_selector, local_state_filter)
        self.is_picnic_over = False

    def reset(self, reachable_positions, state):
        super(SinglePicnicDestinationSelector, self).reset(reachable_positions, state)
        self.is_picnic_over = False

    def get_destination(self, state):
        if self.is_primary_selector:
            if self.is_picnic_over:
                return self.current_destination
            if self.call_init_picnic_pairs_required:
                self.call_init_picnic_pairs_required = False
                self._init_picnic_pairs()
            for destination_selector in self.destination_selectors:
                destination_selector.remove_reached_goals(state)
                if destination_selector._state_filter_update_required():
                    destination_selector._update_current_reachable_goals(state)
            if self.all_destinations_reached(state):
                if self.find_picnic_food:
                    self._update_picnic_pair_ctr()
                self.find_picnic_food = not self.find_picnic_food
                self.assign_destination(state)
        return self.current_destination

    def all_destinations_reached(self, state):
        for i, dest_selector in enumerate(self.destination_selectors):
            if dest_selector.current_destination is None:
                self.destinations_reached[i] = True
            if not self.destinations_reached[i]:
                self.destinations_reached[i] = dest_selector.destination_reached(state)
        destination_reached = all(self.destinations_reached)
        if destination_reached:
            self.destinations_reached = [False for _ in range(len(self.destination_selectors))]
            if self.find_picnic_food:
                self.is_picnic_over = True
        return destination_reached


class RepositionPicnicDestinationSelector(PicnicDestinationSelector):
    def __init__(self, reward_func, agent_idx, goal_ranker, is_primary_selector, local_state_filter=None):
        super().__init__(reward_func, agent_idx, goal_ranker, is_primary_selector, local_state_filter)
        self.finished_picnic = False
        self.finished_picnic_gate = False

    def reset(self, reachable_positions, state):
        super().reset(reachable_positions, state)
        self.finished_picnic = False
        self.finished_picnic_gate = False

    def all_destinations_reached(self, state):
        self.finished_picnic = False
        for i, dest_selector in enumerate(self.destination_selectors):
            if dest_selector.current_destination is None:
                self.destinations_reached[i] = True
            if not self.destinations_reached[i]:
                self.destinations_reached[i] = dest_selector.destination_reached(state)
        destination_reached = all(self.destinations_reached)
        if destination_reached:
            self.destinations_reached = [False for _ in range(len(self.destination_selectors))]
            if not self.find_picnic_food:
                if self.finished_picnic_gate and not self.no_more_picnics:
                    self.finished_picnic = True
                self.finished_picnic_gate = True

        return destination_reached
