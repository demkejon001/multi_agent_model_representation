import numpy as np
from scipy.stats import dirichlet

from tommas.agents.reward_function import RewardFunction
from tommas.env.gridworld_env import GridworldEnv


# Ai = (Ωi, ωi, Ri, γi, πi)
class Agent:
    def __init__(self, agent_id, num_actions, state_function, reward_function, discount):
        self.id = agent_id
        self.num_actions = num_actions
        self.state_function = state_function
        # reward_function has dim (num_agents, num_goals). It is analyzed as (consuming_agent, consumed_goal)
        self.reward_function = reward_function
        self.discount = discount

    def get_rewards(self, collisions):
        reward = 0
        for collision in collisions:
            if collision[0] == -2:  # Player collision
                reward += 0
            elif collision[0] == -1:
                reward += 0
            else:
                reward += self.reward_function[collision[1]][collision[0]]
        return reward

    def action(self, state):
        pass


class RandomAgent(Agent):
    def __init__(self, agent_id, num_actions, alpha):
        super().__init__(agent_id, num_actions, None, None, 1)
        self.policy = dirichlet.rvs(alpha=np.ones(num_actions) * alpha, size=1)[0]

    def action(self, state):
        return np.random.choice(self.num_actions, p=self.policy)


class SingleAgentMDP:
    def __init__(self, env: GridworldEnv, reward_function: RewardFunction):
        env.reset(create_new_world=False)
        self.mdp = dict()
        self.goal_positions = env.world.goal_positions
        self.movement_actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.movement_penalty = reward_function.movement_penalty
        self.wall_collision_penalty = reward_function.wall_collision_penalty
        self.reward_function = reward_function
        self.terminal_position = (0, 0)  # Top corner wall
        self.convert_SingleAgentEnv_to_MDP(env)

    def get_goal_reward(self, goal_index):
        return self.reward_function.goal_rewards[0][int(goal_index)]

    def check_goal_collision(self, state):
        for position, goal_index in self.goal_positions.items():
            if state == position:
                return True
        return False

    def connect_reachable_positions(self, mdp, reachable_positions):
        for position in reachable_positions:
            state_transition = {4: [position, self.movement_penalty]}
            for action, movement_action in enumerate(self.movement_actions):
                new_position = (position[0] + movement_action[0], position[1] + movement_action[1])
                if new_position in mdp:
                    state_transition.update({action: [new_position, self.movement_penalty]})
                else:
                    state_transition.update({action: [position, self.wall_collision_penalty]})
            mdp[position] = state_transition

    def modify_goal_state_transitions(self, mdp):
        for position, goal_index in self.goal_positions.items():
            if position in mdp:
                for action, movement_action in enumerate(self.movement_actions):
                    new_position = (position[0] + movement_action[0], position[1] + movement_action[1])
                    if new_position in mdp:
                        reward = self.get_goal_reward(goal_index)
                        flipped_action = (action + 2) % len(self.movement_actions)
                        mdp[new_position][flipped_action] = [position, reward]
                    mdp[position][action] = [self.terminal_position, 0]
                mdp[position][4] = [self.terminal_position, 0]

    def convert_SingleAgentEnv_to_MDP(self, env: GridworldEnv):
        from tommas.helper_code.navigation import maze_traversal
        state = env.reset(create_new_world=False)
        reachable_positions = maze_traversal(state[0], env.world.agent_positions[0])
        mdp = dict()
        for position in reachable_positions:
            mdp[position] = None

        self.connect_reachable_positions(mdp, reachable_positions)
        self.modify_goal_state_transitions(mdp)

        self.mdp = mdp


class GoalDirectedAgent(Agent):
    def __init__(self, agent_id, num_actions, state_function, reward_function, discount, stochastic=True):
        super(GoalDirectedAgent, self).__init__(agent_id, num_actions, state_function, reward_function, discount)
        self.stochastic = stochastic
        self.policy = None
        self.horizon = 31

    def reset(self, env: GridworldEnv):
        self.value_iteration(env)

    def value_iteration(self, env: GridworldEnv, theta=.0001):
        def get_action_values(s):
            return [mdp[s][a][1] + self.discount * values[mdp[s][a][0]] for a in range(env.num_actions)]

        values = np.zeros(env.dim)
        mdp = SingleAgentMDP(env, self.reward_function).mdp
        for _ in range(self.horizon):
            delta = 0
            for state in mdp:
                max_action_value = max(get_action_values(state))
                delta = max(delta, np.abs(max_action_value - values[state]))
                values[state] = max_action_value
            if delta < theta:
                break

        if self.stochastic:
            policy = np.zeros((*env.dim, self.num_actions))
            for state in mdp:
                action_values = get_action_values(state)
                max_action_value = np.max(action_values)
                action_dist = np.zeros(self.num_actions)
                for action, action_value in enumerate(action_values):
                    if action_value == max_action_value:
                        action_dist[action] = 1
                action_dist = action_dist / np.sum(action_dist)
                policy[state] = action_dist
        else:
            policy = np.zeros(env.dim)
            for state in mdp:
                best_action = np.argmax(get_action_values(state))
                policy[state] = best_action
        self.policy = policy

    def get_agent_position(self, state):
        agent_channel = state[-1]
        return tuple(np.argwhere(agent_channel == 1)[0])

    def action(self, state):
        agent_position = self.get_agent_position(state)
        if self.stochastic:
            return np.random.choice(self.num_actions, size=1, replace=False, p=self.policy[agent_position])[0]
        else:
            return int(self.policy[agent_position])


class MovementPatternAgent(Agent):
    def __init__(self, agent_id, num_actions, movement_pattern=None):
        super(MovementPatternAgent, self).__init__(agent_id, num_actions, None, None, None)
        self.movement_pattern = (movement_pattern if movement_pattern else self.get_random_movement_pattern())
        self.movement_ctr = 0

    def get_random_movement_pattern(self):
        return np.random.choice(self.num_actions, size=3)

    def action(self, state):
        action = self.movement_pattern[self.movement_ctr]
        self.movement_ctr = (self.movement_ctr + 1) % len(self.movement_pattern)
        return action

    def reset(self):
        self.movement_ctr = 0
