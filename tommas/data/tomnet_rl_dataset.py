import pickle
import numpy as np
import torch

from tommas.agents.ikostrikov_a2c import Policy
from tommas.data.agent_ids import get_agent_ids
from tommas.env.gridworld_env import ToMnetRLGridworldEnv
from tommas.env.gridworld_wrappers import FoVWrapper, BlindWrapper, GoalSwapWrapper
from tommas.data.gridworld_dataset import GridworldDataset
from tommas.data.gridworld_trajectory import GridworldTrajectory
from tommas.data.gridworld_datamodule import GridworldDataModule

num_dataset_agents = 1
num_actions = 5
num_goals = 5
fov = 2
train_agent_min_num_episodes = 11
test_agent_min_num_episodes = 11
device = 'cuda' if torch.cuda.is_available() else 'cpu'
shuffle_seed = 42
dataset_seed = 42

rl_agent_filepath = "data/agent_models/tomnet_rl_agents/"
rl_species_1_base_name = 'gridworld_fov_lstm_seed_'
rl_species_2_base_name = 'gridworld_fov_seed_'
rl_species_3_base_name = 'gridworld-blind1_stateless_goalpreference1_seed_'
dataset_filepath = "data/datasets/single_agent_rl/"
file_extension = ".pickle"
dataset_base_name = "tomnet_rl"
dataset_types = ['', 'big']


def load_agent(env, recurrent, filename):
    actor_critic = Policy(env.observation_space.shape, env.action_space, base_kwargs={'recurrent': recurrent})
    print("Loading", filename)
    state_dict = torch.load(filename)
    actor_critic.load_state_dict(state_dict)
    return actor_critic


@torch.no_grad()
def play_episode(tomnet_env, agent_env, agent, goal_permutation):
    trajectory = []
    done = False
    tomnet_obs = tomnet_env.reset()
    agent_obs = agent_env.reset()
    recurrent_hidden_states = torch.zeros(1, agent.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    while not done:
        if goal_permutation is not None:
            agent_obs = agent_obs[goal_permutation]
        obs = torch.tensor(agent_obs).unsqueeze(0).float()
        _, action, _, recurrent_hidden_states = \
            agent.act(obs, recurrent_hidden_states, masks, deterministic=True)
        if str(tomnet_env.env.env.world) != str(agent_env.env.env.env.world):
            raise ValueError('ToMnet Env does not match Agent Env')
        agent_obs, reward, done, info = agent_env.step(action.squeeze().item())
        trajectory.append((tomnet_obs, [reward], [action.squeeze().item()], info))
        tomnet_obs, _, _, _ = tomnet_env.step(action.squeeze().item())
        masks.fill_(0.0 if done else 1.0)
    return trajectory


def get_agent_history(tomnet_env, agent_env, agent, goal_permutation, agent_ids, is_train_agent=True):
    agent_history = []
    num_episodes = train_agent_min_num_episodes if is_train_agent else test_agent_min_num_episodes
    while len(agent_history) < num_episodes:
        trajectory = play_episode(tomnet_env, agent_env, agent, goal_permutation)
        if len(trajectory) > 1:
            agent_history.append(GridworldTrajectory(trajectory, tomnet_env, agent_ids))
    return agent_history


def make_env(env_name, seed, goal_preference):
    if env_name == 'blind':
        return GoalSwapWrapper(BlindWrapper(ToMnetRLGridworldEnv({'seed': seed})), goal_preference)
    elif env_name == 'tomnet':
        return GoalSwapWrapper(ToMnetRLGridworldEnv({'seed': seed}), goal_preference)
    else:
        return GoalSwapWrapper(FoVWrapper(ToMnetRLGridworldEnv({'seed': seed, 'agent_FoV': fov})), goal_preference)


def get_species_1_history(return_info=False):
    agent_seeds = list(range(1, 21)) + list(range(101, 114))
    agent_filename = rl_agent_filepath + rl_species_1_base_name
    best_agent_seeds = get_best_performing_seeds(agent_seeds, agent_filename, recurrent=True, env_name='fov')
    return get_species_history(best_agent_seeds, agent_filename, recurrent=True, env_name='fov',
                               return_info=return_info)


def get_species_2_history(return_info=False):
    agent_seeds = list(range(21, 41)) + list(range(121, 134))
    agent_filename = rl_agent_filepath + rl_species_2_base_name
    best_agent_seeds = get_best_performing_seeds(agent_seeds, agent_filename, recurrent=False, env_name='fov')
    return get_species_history(best_agent_seeds, agent_filename, recurrent=False, env_name='fov',
                               return_info=return_info)


def get_species_3_history(return_info=False):
    agent_seeds = list(range(41, 61)) + list(range(141, 154))
    agent_filename = rl_agent_filepath + rl_species_3_base_name
    best_agent_seeds = get_best_performing_seeds(agent_seeds, agent_filename, recurrent=True, env_name='blind')
    return get_species_history(best_agent_seeds, agent_filename, recurrent=True, env_name='blind',
                               return_info=return_info)


def get_best_performing_seeds(agent_seeds, agent_filename, recurrent, env_name):
    global train_agent_min_num_episodes
    original_train_agent_min_num_episodes = train_agent_min_num_episodes
    train_agent_min_num_episodes = 1
    num_games = 40
    num_best_agent_seeds = 20
    goal_preference = 1
    env_start_seed = 444422
    agents_avg_return = []
    dummy_agent_ids = [0]
    for seed in agent_seeds:
        agent_seed_filename = agent_filename + str(seed) + '.pt'
        agent_env = make_env(env_name, seed, 0)
        agent = load_agent(agent_env, recurrent, agent_seed_filename)
        return_sum = 0
        for env_seed in range(env_start_seed, env_start_seed + num_games):
            agent_env = make_env(env_name, env_seed, goal_preference)
            tomnet_env = make_env('tomnet', env_seed, goal_preference)
            if env_name == 'blind':
                agent_history = get_agent_history(tomnet_env, agent_env, agent, None, dummy_agent_ids)
            else:
                agent_history = get_agent_history(tomnet_env, agent_env, agent, goal_permutation=[0, 1, 2, 3, 4, 5, 6],
                                                  agent_ids=dummy_agent_ids)
            return_sum += agent_history[0].get_return()[0]
        agents_avg_return.append(return_sum / num_games)
    train_agent_min_num_episodes = original_train_agent_min_num_episodes
    best_performing_seeds = np.array(agent_seeds)[np.argsort(agents_avg_return)[-num_best_agent_seeds:]]
    np.random.shuffle(best_performing_seeds)
    return best_performing_seeds


def get_random_seed_offset():
    return np.random.randint(1, 1000000000)


def get_species_history(agent_seeds, agent_filename, recurrent, env_name, return_info=False):
    species_history = []
    agent_ids = []
    agent_info = []
    envs = []
    goal_permutations = np.array(
        [[0, 1, 2, 3, 4, 5, 6], [0, 1, 3, 2, 4, 5, 6], [0, 1, 4, 3, 2, 5, 6], [0, 1, 5, 3, 4, 2, 6]])
    for seed_idx, seed in enumerate(agent_seeds):
        is_training_agent = True
        if seed_idx >= len(agent_seeds) // 2:
            is_training_agent = False
        agent_seed_filename = agent_filename + str(seed) + '.pt'
        agent_env = make_env(env_name, seed, 0)
        agent = load_agent(agent_env, recurrent, agent_seed_filename)
        for goal_preference in range(1, num_goals):
            env_seed = seed * (1000 ** goal_preference) + get_random_seed_offset()
            agent_env = make_env(env_name, env_seed, goal_preference)
            tomnet_env = make_env('tomnet', env_seed, goal_preference)
            goal_permutation = None if env_name == 'blind' else goal_permutations[goal_preference - 1]
            current_agent_id = get_agent_ids(num_agent_ids=1)
            agent_history = get_agent_history(tomnet_env, agent_env, agent, goal_permutation, current_agent_id,
                                              is_training_agent)
            species_history.append(agent_history)
            agent_ids += current_agent_id
            agent_info.append((agent_seed_filename, recurrent, goal_permutation))
            envs.append((agent_env, tomnet_env))
    if return_info:
        return species_history, agent_ids, agent_info, envs
    return species_history, agent_ids


def split_train_test_species_data(data1, data2, data3):
    train_data = data1[:len(data1) // 2] + data2[:len(data2) // 2] + data3[:len(data3) // 2]
    test_data = data1[len(data1) // 2:] + data2[len(data2) // 2:] + data3[len(data3) // 2:]
    return train_data, test_data


def create_train_test_set(big_datasets=False):
    np.random.seed(dataset_seed)
    if big_datasets:
        global train_agent_min_num_episodes, test_agent_min_num_episodes
        train_agent_min_num_episodes = 1000
        test_agent_min_num_episodes = 60
    history_1, agent_ids_1 = get_species_1_history()
    history_2, agent_ids_2 = get_species_2_history()
    history_3, agent_ids_3 = get_species_3_history()
    train_history, test_history = split_train_test_species_data(history_1, history_2, history_3)
    train_history = list(np.concatenate(train_history))
    test_history = list(np.concatenate(test_history))
    train_agent_ids, test_agent_ids = split_train_test_species_data(agent_ids_1, agent_ids_2, agent_ids_3)

    train_dataset = GridworldDataset(train_history)
    test_dataset = GridworldDataset(test_history)
    return train_dataset, test_dataset


class SARLDataModule(GridworldDataModule):
    def __init__(self, dataset_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_type = dataset_type

    def prepare_data(self):
        self.train = load_dataset(is_train_dataset=True, dataset_type=self.dataset_type)
        self.val = load_dataset(is_train_dataset=False, dataset_type=self.dataset_type)


def check_dataset_type(dataset_type):
    if dataset_type not in dataset_types:
        raise ValueError("Invalid dataset type. Expected one of: %s" % dataset_types)


def get_dataset_filename(is_train_dataset, dataset_type):
    check_dataset_type(dataset_type)
    dataset_filename = dataset_filepath + dataset_base_name
    dataset_filename += ("_train" if is_train_dataset else "_test") + dataset_type + file_extension
    return dataset_filename


def save_dataset(dataset, is_train_dataset, dataset_type=''):
    dataset_filename = get_dataset_filename(is_train_dataset, dataset_type)
    file = open(dataset_filename, "wb")
    pickle.dump(dataset, file)
    file.close()


def load_dataset(is_train_dataset=True, dataset_type='') -> GridworldDataset:
    dataset_filename = get_dataset_filename(is_train_dataset, dataset_type)
    file = open(dataset_filename, "rb")
    dataset = pickle.load(file)
    file.close()
    return dataset


def create_and_save_all_datasets(dataset_type='', seed=42):
    global dataset_seed
    dataset_seed = seed
    if dataset_type == "all":
        for dataset_type in dataset_types:
            create_and_save_all_datasets(dataset_type)
    else:
        check_dataset_type(dataset_type)
        if dataset_type == '':
            train, test = create_train_test_set(big_datasets=False)
        elif dataset_type == 'big':
            train, test = create_train_test_set(big_datasets=True)
        else:
            raise ValueError("Dataset Type " + dataset_type + " is not accounted for")
        save_dataset(train, is_train_dataset=True, dataset_type=dataset_type)
        save_dataset(test, is_train_dataset=False, dataset_type=dataset_type)

