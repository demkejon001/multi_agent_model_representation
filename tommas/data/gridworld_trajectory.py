from enum import IntEnum
from typing import List

import numpy as np

from tommas.env.gridworld_env import GridworldEnv


class StateActionPosition(IntEnum):
    STATE = 0
    REWARDS = 1
    ACTIONS = 2
    INFO = 3


class GridworldTrajectory:
    def __init__(self,
                 trajectory: List,
                 env: GridworldEnv,
                 agent_ids: List[int]):
        self.trajectory = trajectory
        self.trajectory_length = len(self.trajectory)
        self.horizon = env.horizon
        self.num_goals = env.num_goals
        self.num_actions = env.num_actions
        self.dim = env.dim
        self.num_agents = env.num_agents
        self.agent_ids = np.array(agent_ids)
        assert self.num_agents == len(self.agent_ids)
        self.keep_goals = False

    def __len__(self):
        return self.trajectory_length

    def agent_id_to_idx(self, agent_id):
        return np.where(self.agent_ids == agent_id)[0][0]

    def spatialise(self, action):
        spatialized_action = np.zeros((self.num_actions - 1, self.dim[0], self.dim[1]))
        ones = np.ones((1, self.dim[0], self.dim[1]))
        return np.insert(spatialized_action, action, ones, axis=0)

    def get_start_token(self, attach_agent_ids=False, remove_actions=False, remove_other_agents=False,
                        remove_other_agents_actions=False):
        num_features = 1 + self.num_goals
        num_features += (1 if remove_other_agents else self.num_agents)
        if not remove_actions:
            num_features += (self.num_actions if remove_other_agents_actions else self.num_actions * self.num_agents)
        if attach_agent_ids:
            num_features += self.num_agents
        return np.ones((num_features, *self.dim))

    # State tensor has [walls, goals, agents' positions, agents' actions, agents' ids]
    def create_state_tensor(self, state, agents_actions, agent_idx, attach_agent_ids=False,
                            remove_actions=False, remove_other_agents=False, remove_other_agents_actions=False):
        # If you are removing the agents then you shouldn't attach agent ids
        assert not (remove_other_agents and attach_agent_ids)
        if self.keep_goals:
            # grabs [walls, goals] at first timestep
            tensor = self.trajectory[0][StateActionPosition.STATE][:1 + self.num_goals]
        else:
            tensor = state[:1 + self.num_goals]  # grabs [walls, goals]
        agent_offset = 1 + self.num_goals

        agent_idxs = np.arange(self.num_agents)
        agent_idxs[0], agent_idxs[agent_idx] = agent_idxs[agent_idx], agent_idxs[0]
        agent_positions = state[agent_offset + agent_idxs]  # grabs [agents' positions]
        if remove_other_agents:
            agent_positions = agent_positions[:1]
            remove_other_agents_actions = True  # removing other agents means removing their actions as well

        state = [tensor, agent_positions]

        if not remove_actions:
            if remove_other_agents_actions:
                action_tensors = np.zeros((self.num_actions, self.dim[0], self.dim[1]))
                action_tensors[agents_actions[agent_idx]] = np.ones((self.dim[0], self.dim[1]))
            else:
                action_tensors = np.zeros((self.num_actions * self.num_agents, self.dim[0], self.dim[1]))
                action_tensors[(agent_idxs * self.num_actions) + agents_actions] = \
                    np.ones((self.num_agents, self.dim[0], self.dim[1]))   # grabs [agents' actions]
            state.append(action_tensors)

        if attach_agent_ids:
            agents_ids_tensor = np.ones((self.num_agents, self.dim[0], self.dim[1]))[range(self.num_agents)] * \
                                self.agent_ids[agent_idxs].reshape(self.num_agents, 1, 1)  # grabs [agents' ids]
            state.append(agents_ids_tensor)

        return np.concatenate(state)

    def get_future_goals_consumed(self, start_index, agent_idx=0, get_all_future_goals=True):
        end_traj = self.trajectory[start_index:]
        goal_consumption = np.zeros(self.num_goals)
        for state_action in end_traj:
            goals_consumed = state_action[StateActionPosition.INFO]
            if bool(goals_consumed):  # if dict isn't empty
                if agent_idx in goals_consumed:
                    goal_consumption[goals_consumed[agent_idx]] = 1
                    if not get_all_future_goals:
                        return goal_consumption
        return goal_consumption

    def get_spatialised_trajectory(self, start_index=0, end_index=None, agent_idx=0, attach_agent_ids=False,
                                   remove_actions=False, remove_other_agents=False, remove_other_agents_actions=False,
                                   add_start_token=False):
        trajectory = []
        if add_start_token:
            trajectory.append(self.get_start_token(attach_agent_ids=attach_agent_ids, remove_actions=remove_actions,
                                                   remove_other_agents=remove_other_agents,
                                                   remove_other_agents_actions=remove_other_agents_actions))
        if end_index is None:
            end_index = self.trajectory_length
        for state_action in self.trajectory[start_index:end_index]:
            state, actions = state_action[StateActionPosition.STATE], state_action[StateActionPosition.ACTIONS]
            trajectory.append(self.create_state_tensor(state, actions, agent_idx, attach_agent_ids=attach_agent_ids,
                                                       remove_actions=remove_actions,
                                                       remove_other_agents=remove_other_agents,
                                                       remove_other_agents_actions=remove_other_agents_actions))
        return np.array(trajectory)

    def get_successor_representations(self, start_index, agent_idx=0):
        gammas = np.array([.5, .9, .99])
        discounts = np.array([.5, .9, .99])
        successor_representations = np.zeros((3, *self.dim))

        end_traj = self.trajectory[start_index + 1:]
        if len(end_traj) == 0:
            end_traj = self.trajectory[start_index:]
        for state_action in end_traj:
            state = state_action[0]
            agent_position = state[1 + self.num_goals + agent_idx]
            for i, discount in enumerate(discounts):
                successor_representations[i] += agent_position * discount
            discounts *= gammas

        # Normalize successor representations
        successor_representations = successor_representations / np.sum(successor_representations, axis=(1, 2),
                                                                       keepdims=True)
        return successor_representations

    def get_state(self, index, agent_idx=0, attach_agent_ids=False, remove_other_agents=False):
        agent_idxs = np.arange(self.num_agents)
        agent_idxs[0], agent_idxs[agent_idx] = agent_idxs[agent_idx], agent_idxs[0]
        agent_offset = 1 + self.num_goals
        if self.keep_goals:
            state = np.concatenate((self.trajectory[0][StateActionPosition.STATE][:1+self.num_goals],
                                    self.trajectory[index][StateActionPosition.STATE][1+self.num_goals:]), axis=0)
        else:
            state = self.trajectory[index][StateActionPosition.STATE]
        agent_positions = state[agent_offset + agent_idxs]  # grabs [agents' positions]
        if remove_other_agents:
            agent_positions = agent_positions[:1]

        if attach_agent_ids:
            agents_ids_tensor = np.ones((self.num_agents, self.dim[0], self.dim[1]))[range(self.num_agents)] * \
                                self.agent_ids[agent_idxs].reshape(self.num_agents, 1, 1)
            return np.concatenate((state[:agent_offset], agent_positions, agents_ids_tensor), axis=0)
        else:
            return np.concatenate((state[:agent_offset], agent_positions), axis=0)

    def get_action(self, index, agent_idx=0):
        return self.trajectory[index][StateActionPosition.ACTIONS][agent_idx]

    def get_independent_agent_features(self, remove_actions=False):
        action_offset = 1 + self.num_goals + self.num_agents
        ia_features = list(range(1 + self.num_goals + 1))
        if not remove_actions:
            ia_features = ia_features + list(range(action_offset, action_offset + self.num_actions))
        return ia_features

    def get_return(self):
        ret = np.zeros(self.num_agents)
        for _, reward, _, _ in self.trajectory:
            ret += reward
        return ret

    def get_unspatialised_state(self, spatialised_state):
        num_gridworld_features = 1 + self.num_goals + self.num_agents
        state = spatialised_state[0:num_gridworld_features]
        action_vector = spatialised_state[num_gridworld_features:, 0, 0]
        for action, element in enumerate(action_vector):
            if int(element) == 1:
                return state, action
        raise TypeError('No action present in spatialised_state')

    def print_as_gridworld(self, state):
        gridworld_state = (state[:1] * -1)
        for i in range(self.num_agents):
            agent_position = 1 + self.num_goals + i
            print('agent pos', agent_position)
            gridworld_state += state[agent_position:agent_position + 1] * (50 + i)
        for i in range(self.num_goals):
            goal_position = 1 + i
            print('goal pos', goal_position)
            gridworld_state += state[goal_position:goal_position + 1] * (1 + i)
        print(gridworld_state.astype(np.int).__repr__())

    def __eq__(self, other):
        if not isinstance(other, GridworldTrajectory):
            return False
        if self.trajectory_length != other.trajectory_length:
            return False
        if self.horizon != other.horizon:
            return False
        if self.num_goals != other.num_goals:
            return False
        if self.num_actions != other.num_actions:
            return False
        if self.dim != other.dim:
            return False
        if self.num_agents != other.num_agents:
            return False
        if not np.array_equal(self.agent_ids.sort(), other.agent_ids.sort()):
            return False
        for step_a, step_b in zip(self.trajectory, other.trajectory):
            if not np.array_equal(step_a[0], step_b[0]):
                return False
            if not np.array_equal(step_a[1], step_b[1]):
                return False
            if not np.array_equal(step_a[2], step_b[2]):
                return False
            if not step_a[3] == step_b[3]:
                return False
        return True


class UnspatialisedGridworldTrajectory:
    def __init__(self,
                 trajectory: List,
                 env: GridworldEnv,
                 agent_ids: List[int]):
        self.trajectory = trajectory
        self.trajectory_length = len(self.trajectory)
        self.horizon = env.horizon
        self.num_goals = env.num_goals
        self.num_actions = env.num_actions
        self.dim = env.dim
        self.num_agents = env.num_agents
        self.agent_ids = np.array(agent_ids)
        assert self.num_agents == len(self.agent_ids)

    def __len__(self):
        return self.trajectory_length

    def agent_id_to_idx(self, agent_id):
        return np.where(self.agent_ids == agent_id)[0][0]

    def get_start_state_token(self):
        num_features = 1 + self.num_goals + self.num_agents
        return np.full((num_features, *self.dim), -1)

    def get_start_action_token(self):
        num_features = self.num_agents * self.num_actions
        return np.ones(num_features) * -1

    def get_future_goals_consumed(self, start_index, agent_idx=0, get_all_future_goals=True):
        end_traj = self.trajectory[start_index:]
        goal_consumption = np.zeros(self.num_goals)
        for state_action in end_traj:
            goals_consumed = state_action[StateActionPosition.INFO]
            if bool(goals_consumed):  # if dict isn't empty
                if agent_idx in goals_consumed:
                    goal_consumption[goals_consumed[agent_idx]] = 1
                    if not get_all_future_goals:
                        return goal_consumption
        return goal_consumption

    def get_trajectory(self, start_index=0, end_index=None, agent_idx=0, add_start_token=False):
        trajectory = []
        actions = []
        if end_index is None:
            end_index = self.trajectory_length

        if add_start_token:
            trajectory.append(self.get_start_state_token())
            actions.append(self.get_start_action_token())

        for idx in range(start_index, end_index):
            trajectory.append(self.get_state(idx, agent_idx))
            actions.append(self.get_joint_action(idx, agent_idx))
        return np.array(trajectory), np.array(actions)

    def get_successor_representations(self, start_index, agent_idx=0):
        gammas = np.array([.5, .9, .99])
        discounts = np.array([.5, .9, .99])
        successor_representations = np.zeros((3, *self.dim))

        end_traj = self.trajectory[start_index + 1:]
        if len(end_traj) == 0:
            end_traj = self.trajectory[start_index:]
        for state_action in end_traj:
            state = state_action[0]
            agent_position = state[1 + self.num_goals + agent_idx]
            for i, discount in enumerate(discounts):
                successor_representations[i] += agent_position * discount
            discounts *= gammas

        # Normalize successor representations
        successor_representations = successor_representations / np.sum(successor_representations, axis=(1, 2),
                                                                       keepdims=True)
        return successor_representations

    def get_state(self, index, agent_idx=0):
        agent_idxs = np.arange(self.num_agents)
        agent_idxs[0], agent_idxs[agent_idx] = agent_idxs[agent_idx], agent_idxs[0]
        agent_offset = 1 + self.num_goals
        state = self.trajectory[index][StateActionPosition.STATE]
        agent_positions = state[agent_offset + agent_idxs]  # grabs [agents' positions]
        return np.concatenate((state[:agent_offset], agent_positions), axis=0)

    def get_action(self, index, agent_idx=0):
        return self.trajectory[index][StateActionPosition.ACTIONS][agent_idx]

    def get_joint_action(self, index, agent_idx=0):
        agent_idxs = list(range(self.num_agents))
        agent_idxs[0], agent_idxs[agent_idx] = agent_idxs[agent_idx], agent_idxs[0]
        actions = np.array(self.trajectory[index][StateActionPosition.ACTIONS])
        # rearrange actions so agent_idx is starting action
        rearranged_actions = actions[agent_idxs]
        ja_tensor = np.zeros((actions.size, self.num_actions))
        ja_tensor[np.arange(actions.size), rearranged_actions] = 1
        return np.concatenate(ja_tensor)

    def get_return(self):
        ret = np.zeros(self.num_agents)
        for _, reward, _, _ in self.trajectory:
            ret += reward
        return ret

    def get_unspatialised_state(self, spatialised_state):
        num_gridworld_features = 1 + self.num_goals + self.num_agents
        state = spatialised_state[0:num_gridworld_features]
        action_vector = spatialised_state[num_gridworld_features:, 0, 0]
        for action, element in enumerate(action_vector):
            if int(element) == 1:
                return state, action
        raise TypeError('No action present in spatialised_state')

    def print_as_gridworld(self, state):
        gridworld_state = (state[:1] * -1)
        for i in range(self.num_agents):
            agent_position = 1 + self.num_goals + i
            print('agent pos', agent_position)
            gridworld_state += state[agent_position:agent_position + 1] * (50 + i)
        for i in range(self.num_goals):
            goal_position = 1 + i
            print('goal pos', goal_position)
            gridworld_state += state[goal_position:goal_position + 1] * (1 + i)
        print(gridworld_state.astype(np.int).__repr__())
