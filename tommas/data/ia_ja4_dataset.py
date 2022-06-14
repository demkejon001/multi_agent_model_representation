import math
import numpy as np

from tommas.agents.create_handcodedagents import create_agent
from tommas.agents.reward_function import get_cooperative_reward_functions
from tommas.env.gridworld_env import GridworldEnv
from tommas.env.gridworld_wrappers import StationaryAgentsEarlyTermination
from tommas.data.gridworld_trajectory import GridworldTrajectory
from tommas.data.dataset_base import save_dataset as base_save_dataset, \
    get_dataset_filepaths as base_get_dataset_filepaths
from tommas.data.cooperative_rewards_dataset import reassign_agent_indices


# Environment parameters
num_agents_per_episode = 4
num_goals = 12
horizon = 50
dim = (21, 21)
num_actions = 5
max_num_walls = 6

# Dataset parameters
num_train_dataset_agents = 1000
num_test_dataset_agents = 100
min_num_episodes_per_agent = 21

dataset_dirpath = "data/datasets/ia_ja4/"
dataset_base_name = "ia_ja"


def play_episode(env, agents, create_new_world=True):
    trajectory = []
    state = env.reset(create_new_world=create_new_world)
    for agent_idx, agent in enumerate(agents):
        agent.reset(state, agent_idx)
    done = False
    while not done:
        actions = [agent.action(state) for agent in agents]
        new_state, reward, done, info = env.step(actions)
        trajectory.append((state, reward, actions, info))
        state = new_state
    agent_ids = [agent.id for agent in agents]
    if len(trajectory) <= 1:
        return play_episode(env, agents)
    return GridworldTrajectory(trajectory, env, agent_ids)


def get_random_goal_rewards():
    goal_rewards = np.random.dirichlet(alpha=np.ones(num_goals), size=1)[0]
    num_negative_goals = np.random.choice(range(0, 4), size=None, p=[.4, .3, .2, .1]) * 3 + np.random.randint(0, 3)
    goal_rewards[np.argsort(goal_rewards)[:num_negative_goals]] *= -1
    return goal_rewards * 2


def get_random_coop_reward_funcs():
    goal_rewards = get_random_goal_rewards()
    return get_cooperative_reward_functions(num_agents_per_episode, goal_rewards), goal_rewards


def get_random_env(reward_funcs):
    env_seed = np.random.randint(0, 1e10)
    env = GridworldEnv(num_agents_per_episode, reward_funcs, num_goals, horizon, dim, max_num_walls=max_num_walls,
                       seed=env_seed)
    return StationaryAgentsEarlyTermination(env)


def collect_no_group_data(num_agents, dataset_filename):
    """
    Random sample without replacement from following agents:
    ['highest', ]
    ['highest-static_circle=2']
    ['highest-static_circle=3']
    ['highest-static_circle=4']
    ['highest-static_circle=5']
    ['highest-static_circle=6']
    ['highest-static_circle=8']
    ['highest-static_circle=9']
    ['highest-static_diamond=3']
    ['highest-static_diamond=5']
    ['highest-static_diamond=4']
    ['highest-static_diamond=6']
    ['highest-static_diamond=7']
    ['highest-static_diamond=8']
    ['highest-static_diamond=9']
    ['closest_distance', ]
    ['discount_distance=0.9']
    ['discount_distance=0.95']
    ['discount_distance=0.9-static_diamond=10']
    ['discount_distance=0.95-static_square=9']
    """
    def add_agent_kwargs(goal_ranker_type, goal_ranker_args=None, state_filter_type=None, state_filter_args=None,
                         is_collaborative=False):
        nonlocal create_agent_idx
        create_agent_kwargs[create_agent_idx] = {"goal_ranker_type": goal_ranker_type,
                                                 "goal_ranker_args": goal_ranker_args,
                                                 "state_filter_type": state_filter_type,
                                                 "state_filter_args": state_filter_args,
                                                 "is_collaborative": is_collaborative}
        create_agent_idx += 1

    create_agent_idx = 0
    create_agent_kwargs = dict()
    static_filter_types_and_args = [(None, None)]
    for radius in [2, 3, 4, 5, 6, 8, 9]:
        static_filter_types_and_args.append(("static_circle", (radius,)))
    for radius in [3, 4, 5, 6, 7, 8, 9]:
        static_filter_types_and_args.append(("static_diamond", (radius,)))
    for filter_type, filter_args in static_filter_types_and_args:
        add_agent_kwargs("highest", (False,), filter_type, filter_args, False)

    add_agent_kwargs("closest_distance", is_collaborative=False)
    add_agent_kwargs(goal_ranker_type="discount_distance", goal_ranker_args=(.95, ), is_collaborative=False)
    add_agent_kwargs(goal_ranker_type="discount_distance", goal_ranker_args=(.9,), is_collaborative=False)
    add_agent_kwargs(goal_ranker_type="discount_distance", goal_ranker_args=(.9,), state_filter_type="static_diamond",
                     state_filter_args=(10,), is_collaborative=False)
    add_agent_kwargs(goal_ranker_type="discount_distance", goal_ranker_args=(.95,), state_filter_type="static_square",
                     state_filter_args=(9,), is_collaborative=False)

    file_idx = 0
    agent_behaviors = dict()
    for _ in range(num_agents):
        agent_func_indices = np.random.choice(len(create_agent_kwargs), size=num_agents_per_episode, replace=False)
        agents_kwargs = [create_agent_kwargs[agent_func_idx] for agent_func_idx in agent_func_indices]
        reward_funcs, goal_rewards = get_random_coop_reward_funcs()
        agents = []
        for i, kwargs in enumerate(agents_kwargs):
            group_agents, group_ids, agent_names = create_agent(reward_funcs[i: i+1], **kwargs)
            agents += group_agents
            agent_behaviors[group_ids[0]] = agent_names[0]
        reassign_agent_indices(agents)
        env = get_random_env(reward_funcs)
        base_save_dataset([play_episode(env, agents) for _ in range(min_num_episodes_per_agent)],
                          dataset_dirpath, dataset_filename + str(file_idx))
        file_idx += 1
    return agent_behaviors


def collect_picnic_data(num_agents, dataset_filename):
    def add_picnic_agent_kwargs(goal_ranker_type, goal_ranker_args=None):
        nonlocal create_agent_idx
        create_agent_kwargs[create_agent_idx] = {"goal_ranker_type": goal_ranker_type,
                                                 "goal_ranker_args": goal_ranker_args,
                                                 "agent_type": "picnic"}
        create_agent_idx += 1

    create_agent_idx = 0
    create_agent_kwargs = dict()

    file_idx = 0
    agent_behaviors = dict()
    add_picnic_agent_kwargs(goal_ranker_type="highest", goal_ranker_args=(True,))
    add_picnic_agent_kwargs(goal_ranker_type="highest", goal_ranker_args=(False,))
    add_picnic_agent_kwargs(goal_ranker_type="closest_distance", goal_ranker_args=(True,))
    add_picnic_agent_kwargs(goal_ranker_type="discount_distance", goal_ranker_args=(.95,))
    add_picnic_agent_kwargs(goal_ranker_type="discount_distance", goal_ranker_args=(.9,))
    for _ in range(math.ceil(num_agents / len(create_agent_kwargs))):
        for kwargs in create_agent_kwargs.values():
            reward_funcs, goal_rewards = get_random_coop_reward_funcs()
            picnic_agents, picnic_agent_ids, picnic_agent_names = create_agent(reward_funcs, **kwargs)
            env = get_random_env(reward_funcs)
            for agent_id, name in zip(picnic_agent_ids, picnic_agent_names):
                agent_behaviors[agent_id] = name
            base_save_dataset([play_episode(env, picnic_agents) for _ in range(min_num_episodes_per_agent)],
                              dataset_dirpath, dataset_filename + str(file_idx))
            file_idx += 1
    return agent_behaviors


def create_ia_dataset(seed, num_agents, dataset_filename):
    np.random.seed(seed)
    agent_behaviors = collect_no_group_data(num_agents, dataset_filename)
    base_save_dataset(agent_behaviors, dataset_dirpath, dataset_filename + "_behaviors")


def create_ja_dataset(seed, num_agents, dataset_filename):
    np.random.seed(seed)
    agent_behaviors = collect_picnic_data(num_agents, dataset_filename)
    base_save_dataset(agent_behaviors, dataset_dirpath, dataset_filename + "_behaviors")


def get_dataset_filename(is_ia: bool, is_train_dataset: bool):
    dataset_filename = ""
    if is_ia:
        dataset_filename += "ia"
    else:
        dataset_filename += "ja"
    dataset_filename += str(num_agents_per_episode)
    dataset_type = '_test'
    if is_train_dataset:
        dataset_type = '_train'
    return dataset_filename + dataset_type


def create_and_save_all_datasets(seed=42):
    for is_ia in [True, False]:
        for is_training_dataset in [True, False]:
            num_dataset_agents = num_train_dataset_agents if is_training_dataset else num_test_dataset_agents
            filename = get_dataset_filename(is_ia, is_training_dataset)
            if is_ia:
                create_ia_dataset(seed, num_dataset_agents, filename)
            else:
                create_ja_dataset(seed, num_dataset_agents, filename)
            seed += 1


def get_dataset_filepaths(is_ia, is_train_dataset):
    filename = get_dataset_filename(is_ia=is_ia, is_train_dataset=is_train_dataset)
    return base_get_dataset_filepaths(dataset_dirpath, filename)
