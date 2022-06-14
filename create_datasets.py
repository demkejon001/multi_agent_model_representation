import argparse
import parser
from typing import Callable, Optional, Union

from tommas.data import debug_dataset, random_policy_dataset, cooperative_rewards_dataset, ia_ja2_dataset, \
    ia_ja4_dataset, iterative_action_dataset, iterative_action_fake_ja_dataset, cooperative_gridworld_dataset, \
    iterative_action_gridworld_dataset, multi_strategy_gridworld_dataset


dataset_creation_data = dict()


def add_dataset(argparse_name: str, create_func: Callable, create_func_kwargs: Optional[dict] = None,
                verbose_name: Optional[str] = None):
    if argparse_name in dataset_creation_data:
        raise ValueError(f"argparse_name {argparse_name} has already been added. Please choose another name")
    if create_func_kwargs is None:
        create_func_kwargs = dict()
    dataset_creation_data[argparse_name] = (create_func, create_func_kwargs, verbose_name)


# add_dataset(argparse_name="tomnet_rl", fetch_method=, create_func=tomnet_rl_dataset.create_and_save_all_datasets,
#             create_func_kwargs={"dataset_type": ""}, verbose_name="ToMnet RL")
# add_dataset(argparse_name="tomnet_rl_big", create_func=tomnet_rl_dataset.create_and_save_all_datasets,
#             create_func_kwargs={"dataset_type": "big"}, verbose_name="ToMnet RL (Big)")
add_dataset(argparse_name="debug", create_func=debug_dataset.create_dataset,
            create_func_kwargs={"single_agent": False}, verbose_name="Debug")
# add_dataset(argparse_name="debug_single", create_func=debug_dataset.create_dataset,
#             create_func_kwargs={"single_agent": True}, verbose_name="Debug (Single Agent)")
# add_dataset(argparse_name="tomnet_goal_directed", create_func=tomnet_goal_directed_dataset.create_and_save_all_datasets,
#             verbose_name="ToMnet Goal Directed")
add_dataset(argparse_name="random_policy", create_func=random_policy_dataset.create_dataset,
            verbose_name="Random Policy")
add_dataset(argparse_name="random_policy_single", create_func=random_policy_dataset.create_single_agent_dataset,
            verbose_name="Random Policy (Single Agent)")
# add_dataset(argparse_name="coop", create_func=cooperative_rewards_dataset.create_and_save_all_datasets,
#             create_func_kwargs={"dataset_type": 'coop'}, verbose_name="Cooperative Reward")
# add_dataset(argparse_name="coop_small", create_func=cooperative_rewards_dataset.create_and_save_all_datasets,
#             create_func_kwargs={"dataset_type": 'coop_small'}, verbose_name="Cooperative Reward (Small)")
# add_dataset(argparse_name="coop_collaborative", create_func=cooperative_rewards_dataset.create_and_save_all_datasets,
#             create_func_kwargs={"dataset_type": 'coop_collaborative'},
#             verbose_name="Cooperative Reward (Collaborative)")
# add_dataset(argparse_name="coop_collaborative_small", create_func=cooperative_rewards_dataset.create_and_save_all_datasets,
#             create_func_kwargs={"dataset_type": 'coop_collaborative_small'},
#             verbose_name="Cooperative Reward (Collaborative, Small)")
add_dataset(argparse_name="ia_ja2", create_func=ia_ja2_dataset.create_and_save_all_datasets,
            verbose_name="IA/JA (2 agents)")
add_dataset(argparse_name="ia_ja4", create_func=ia_ja4_dataset.create_and_save_all_datasets,
            verbose_name="IA/JA (4 agents)")

add_dataset(argparse_name="action_ia2", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_strategy": 400},
                                "num_test_agent_types": {"mixed_strategy": 80},
                                "num_opponents": 1, "is_ia": True},
            verbose_name="Iterative Action IA (2 agents)")
add_dataset(argparse_name="action_ja2", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mirror": 400},
                                "num_test_agent_types": {"mirror": 80},
                                "num_opponents": 1, "is_ia": False},
            verbose_name="Iterative Action JA (2 agents)")
add_dataset(argparse_name="action_ia3", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_strategy": 400},
                                "num_test_agent_types": {"mixed_strategy": 80},
                                "num_opponents": 2, "is_ia": True},
            verbose_name="Iterative Action IA (3 agents)")
add_dataset(argparse_name="action_ja3", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mirror": 400},
                                "num_test_agent_types": {"mirror": 80},
                                "num_opponents": 2, "is_ia": False},
            verbose_name="Iterative Action JA (3 agents)")
add_dataset(argparse_name="action_ia4", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_strategy": 800},
                                "num_test_agent_types": {"mixed_strategy": 80},
                                "num_opponents": 3, "is_ia": True},
            verbose_name="Iterative Action IA (4 agents)")
add_dataset(argparse_name="action_ja4", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mirror": 800},
                                "num_test_agent_types": {"mirror": 80},
                                "num_opponents": 3, "is_ia": False},
            verbose_name="Iterative Action JA (4 agents)")
add_dataset(argparse_name="action_ia5", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_strategy": 1600},
                                "num_test_agent_types": {"mixed_strategy": 80},
                                "num_opponents": 4, "is_ia": True},
            verbose_name="Iterative Action IA (5 agents)")
add_dataset(argparse_name="action_ja5", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mirror": 1600},
                                "num_test_agent_types": {"mirror": 80},
                                "num_opponents": 4, "is_ia": False},
            verbose_name="Iterative Action JA (5 agents)")
add_dataset(argparse_name="action_ia6", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_strategy": 3200},
                                "num_test_agent_types": {"mixed_strategy": 80},
                                "num_opponents": 5, "is_ia": True},
            verbose_name="Iterative Action IA (6 agents)")
add_dataset(argparse_name="action_ja6", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mirror": 3200},
                                "num_test_agent_types": {"mirror": 80},
                                "num_opponents": 5, "is_ia": False},
            verbose_name="Iterative Action JA (6 agents)")
add_dataset(argparse_name="action_ia7", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_strategy": 3200},
                                "num_test_agent_types": {"mixed_strategy": 80},
                                "num_opponents": 6, "is_ia": True},
            verbose_name="Iterative Action IA (7 agents)")
add_dataset(argparse_name="action_ja7", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mirror": 3200},
                                "num_test_agent_types": {"mirror": 80},
                                "num_opponents": 6, "is_ia": False},
            verbose_name="Iterative Action JA (7 agents)")
add_dataset(argparse_name="action_ia8", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_strategy": 4000},
                                "num_test_agent_types": {"mixed_strategy": 80},
                                "num_opponents": 7, "is_ia": True},
            verbose_name="Iterative Action IA (8 agents)")
add_dataset(argparse_name="action_ja8", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mirror": 4000},
                                "num_test_agent_types": {"mirror": 80},
                                "num_opponents": 7, "is_ia": False},
            verbose_name="Iterative Action JA (8 agents)")

add_dataset(argparse_name="action_pattern_ia2", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"action_pattern_and_mixed_strategy": 200},
                                "num_test_agent_types": {"action_pattern_and_mixed_strategy": 20},
                                "num_opponents": 1, "is_ia": True, "tags": "pattern"},
            verbose_name="Iterative Action Pattern IA (2 agents)")
add_dataset(argparse_name="action_pattern_ja2", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"action_pattern_and_mirror": 200},
                                "num_test_agent_types": {"action_pattern_and_mirror": 20},
                                "num_opponents": 1, "is_ia": False, "tags": "pattern"},
            verbose_name="Iterative Action Pattern JA (2 agents)")

add_dataset(argparse_name="action_simple_ia2", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"action_pattern": 4000},
                                "num_test_agent_types": {"action_pattern": 80},
                                "num_opponents": 1, "is_ia": True, "tags": "simple", "horizon": 40},
            verbose_name="Iterative Simple Action IA (2 agents)")
# add_dataset(argparse_name="action_simple_ia4", create_func=iterative_action_dataset.create_dataset,
#             create_func_kwargs={"num_train_agent_types": {"action_pattern": 400},
#                                 "num_test_agent_types": {"action_pattern": 40},
#                                 "num_opponents": 3, "is_ia": True, "tags": "simple"},
#             verbose_name="Iterative Simple Action IA (4 agents)")
# add_dataset(argparse_name="action_simple_ia8", create_func=iterative_action_dataset.create_dataset,
#             create_func_kwargs={"num_train_agent_types": {"action_pattern": 800},
#                                 "num_test_agent_types": {"action_pattern": 40},
#                                 "num_opponents": 7, "is_ia": True, "tags": "simple"},
#             verbose_name="Iterative Simple Action IA (8 agents)")

add_dataset(argparse_name="action_mixed_trigger_ja2", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 800},
                                "num_test_agent_types": {"mixed_trigger_pattern": 80},
                                "num_opponents": 1, "is_ia": False, "tags": "mixed_trigger"},
            verbose_name="Iterative Action Mixed Trigger Pattern JA (2 agents)")
add_dataset(argparse_name="action_mixed_trigger_ja3", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 800},
                                "num_test_agent_types": {"mixed_trigger_pattern": 80},
                                "num_opponents": 2, "is_ia": False, "tags": "mixed_trigger"},
            verbose_name="Iterative Action Mixed Trigger Pattern JA (3 agents)")
add_dataset(argparse_name="action_mixed_trigger_ja4", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 1600},
                                "num_test_agent_types": {"mixed_trigger_pattern": 80},
                                "num_opponents": 3, "is_ia": False, "tags": "mixed_trigger"},
            verbose_name="Iterative Action Mixed Trigger Pattern JA (4 agents)")
add_dataset(argparse_name="action_mixed_trigger_ja5", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 3200},
                                "num_test_agent_types": {"mixed_trigger_pattern": 80},
                                "num_opponents": 4, "is_ia": False, "tags": "mixed_trigger"},
            verbose_name="Iterative Action Mixed Trigger Pattern JA (5 agents)")
add_dataset(argparse_name="action_mixed_trigger_ja6", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 6400},
                                "num_test_agent_types": {"mixed_trigger_pattern": 80},
                                "num_opponents": 5, "is_ia": False, "tags": "mixed_trigger"},
            verbose_name="Iterative Action Mixed Trigger Pattern JA (6 agents)")
add_dataset(argparse_name="action_mixed_trigger_ja7", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 6400},
                                "num_test_agent_types": {"mixed_trigger_pattern": 80},
                                "num_opponents": 6, "is_ia": False, "tags": "mixed_trigger"},
            verbose_name="Iterative Action Mixed Trigger Pattern JA (7 agents)")
add_dataset(argparse_name="action_mixed_trigger_ja8", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 8000},
                                "num_test_agent_types": {"mixed_trigger_pattern": 80},
                                "num_opponents": 7, "is_ia": False, "tags": "mixed_trigger"},
            verbose_name="Iterative Action Mixed Trigger Pattern JA (8 agents)")

add_dataset(argparse_name="action_fake_mixed_trigger_ja2", create_func=iterative_action_fake_ja_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 800},
                                "num_test_agent_types": {"mixed_trigger_pattern": 80},
                                "num_opponents": 1, "is_ia": False, "tags": "fake_mixed_trigger"},
            verbose_name="Iterative Action Fake Mixed Trigger Pattern JA (2 agents)")
add_dataset(argparse_name="action_fake_mixed_trigger_ja7", create_func=iterative_action_fake_ja_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 2400},
                                "num_test_agent_types": {"mixed_trigger_pattern": 80},
                                "num_opponents": 6, "is_ia": False, "tags": "fake_mixed_trigger"},
            verbose_name="Iterative Action Fake Mixed Trigger Pattern JA (7 agents)")

add_dataset(argparse_name="action_multi_strat_ja4", create_func=iterative_action_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mirror": 800, "action_pattern": 800, "wsls": 800,
                                                          "grim_trigger": 800, "mixed_trigger_pattern": 1600},
                                "num_test_agent_types": {"mirror": 20, "action_pattern": 20, "wsls": 20,
                                                         "grim_trigger": 20, "mixed_trigger_pattern": 20},
                                "num_opponents": 3, "is_ia": False, "tags": "multi_strat"},
            verbose_name="Iterative Action Multi Strategy JA (4 agents)")

for i in [2]:
    num_opponents = i - 1
    add_dataset(argparse_name=f"action_mirror_ja{i}", create_func=iterative_action_dataset.create_dataset,
                create_func_kwargs={"num_train_agent_types": {"mirror": 400},
                                    "num_test_agent_types": {"mirror": 20},
                                    "num_opponents": num_opponents, "is_ia": False, "tags": "mirror"},
                verbose_name=f"Iterative Action Mirror JA ({i} agents)")
    add_dataset(argparse_name=f"action_action_pattern_ja{i}", create_func=iterative_action_dataset.create_dataset,
                create_func_kwargs={"num_train_agent_types": {"action_pattern": 400},
                                    "num_test_agent_types": {"action_pattern": 20},
                                    "num_opponents": num_opponents, "is_ia": False, "tags": "action_pattern"},
                verbose_name=f"Iterative Action Pattern JA ({i} agents)")
    add_dataset(argparse_name=f"action_wsls_ja{i}", create_func=iterative_action_dataset.create_dataset,
                create_func_kwargs={"num_train_agent_types": {"wsls": 400},
                                    "num_test_agent_types": {"wsls": 20},
                                    "num_opponents": num_opponents, "is_ia": False, "tags": "wsls"},
                verbose_name=f"Iterative Action WSLS JA ({i} agents)")
    add_dataset(argparse_name=f"action_grim_trigger_ja{i}", create_func=iterative_action_dataset.create_dataset,
                create_func_kwargs={"num_train_agent_types": {"grim_trigger": 400},
                                    "num_test_agent_types": {"grim_trigger": 20},
                                    "num_opponents": num_opponents, "is_ia": False, "tags": "grim_trigger"},
                verbose_name=f"Iterative Action Grim Trigger JA ({i} agents)")
    add_dataset(argparse_name=f"action_mixed_trigger_pattern_ja{i}", create_func=iterative_action_dataset.create_dataset,
                create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 800},
                                    "num_test_agent_types": {"mixed_trigger_pattern": 20},
                                    "num_opponents": num_opponents, "is_ia": False, "tags": "mixed_trigger_pattern"},
                verbose_name=f"Iterative Action Mixed Trigger Pattern JA ({i} agents)")


for i in [4]:
    num_opponents = i - 1
    add_dataset(argparse_name=f"action_mirror_ja{i}", create_func=iterative_action_dataset.create_dataset,
                create_func_kwargs={"num_train_agent_types": {"mirror": 800},
                                    "num_test_agent_types": {"mirror": 80},
                                    "num_opponents": num_opponents, "is_ia": False, "tags": "mirror"},
                verbose_name=f"Iterative Action Mirror JA ({i} agents)")
    add_dataset(argparse_name=f"action_wsls_ja{i}", create_func=iterative_action_dataset.create_dataset,
                create_func_kwargs={"num_train_agent_types": {"wsls": 800},
                                    "num_test_agent_types": {"wsls": 80},
                                    "num_opponents": num_opponents, "is_ia": False, "tags": "wsls"},
                verbose_name=f"Iterative Action WSLS JA ({i} agents)")
    add_dataset(argparse_name=f"action_grim_trigger_ja{i}", create_func=iterative_action_dataset.create_dataset,
                create_func_kwargs={"num_train_agent_types": {"grim_trigger": 800},
                                    "num_test_agent_types": {"grim_trigger": 80},
                                    "num_opponents": num_opponents, "is_ia": False, "tags": "grim_trigger"},
                verbose_name=f"Iterative Action Grim Trigger JA ({i} agents)")
    add_dataset(argparse_name=f"action_mixed_trigger_pattern_ja{i}", create_func=iterative_action_dataset.create_dataset,
                create_func_kwargs={"num_train_agent_types": {"mixed_trigger_pattern": 1600},
                                    "num_test_agent_types": {"mixed_trigger_pattern": 160},
                                    "num_opponents": num_opponents, "is_ia": False, "tags": "mixed_trigger_pattern"},
                verbose_name=f"Iterative Action Mixed Trigger Pattern JA ({i} agents)")

add_dataset(argparse_name="gridworld", create_func=cooperative_gridworld_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"independent": 1024, "collaborative": 1024},
                                "num_test_agent_types": {"independent": 24, "collaborative": 24}})

add_dataset(argparse_name="iterative_action_gridworld", create_func=iterative_action_gridworld_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"collaborative": 480, "independent": 480},
                                "num_test_agent_types": {"collaborative": 24, "independent": 24}})

add_dataset(argparse_name="multi_strategy_gridworld", create_func=multi_strategy_gridworld_dataset.create_dataset,
            create_func_kwargs={"num_train_agent_types": {"mirror": 800, "wsls": 800, "grim_trigger": 800,
                                                          "mixed_trigger_pattern": 1600},
                                "num_test_agent_types": {"mirror": 80, "wsls": 80, "grim_trigger": 80,
                                                          "mixed_trigger_pattern": 160}})


def get_args():
    parser = argparse.ArgumentParser(description='Creates datasets and saves them')
    parser.add_argument('--seed', type=int, default=42,
                        help='dataset creation seed for reproducibility (default: 42)')

    for dataset in dataset_creation_data:
        verbose_name = get_dataset_verbose_name(dataset)
        parser.add_argument("--" + dataset, action='store_true', default=False,
                            help=f'creates the {verbose_name} dataset (default: False)')

    parsed_args = parser.parse_args()
    return parsed_args


def get_dataset_verbose_name(dataset):
    verbose_name = dataset_creation_data[dataset][2]
    if verbose_name is None:
        verbose_name = dataset
    return verbose_name


def create_datasets(args):
    def confirm_answer():
        if answer.lower() in ['y', 'yes']:
            print("Creating datasets")
            return True
        else:
            print("Cancelling dataset creation")
            return False

    def get_answer():
        if len(datasets_to_create) <= 0:
            raise parser.ParserError("Must select a dataset to create")
        print("Are you sure you want to create and potentially overwrite the following dataset?")
        for dataset_name in datasets_to_create:
            print(' -', dataset_name)
        return input("[y/N] \n")

    datasets_to_create = []
    namespace = vars(args)
    for dataset, should_create in namespace.items():
        if dataset is "seed":
            continue
        if should_create:
            verbose_name = get_dataset_verbose_name(dataset)
            datasets_to_create.append(verbose_name)

    answer = get_answer()
    if confirm_answer():
        seed = args.seed

        for dataset, should_create in namespace.items():
            if dataset is "seed":
                continue
            if should_create:
                verbose_name = get_dataset_verbose_name(dataset)
                print(f'Creating {verbose_name} Dataset')
                create_func, func_kwargs, _ = dataset_creation_data[dataset]
                create_func(seed=seed, **func_kwargs)


def main():
    args = get_args()
    create_datasets(args)


if __name__ == "__main__":
    main()

