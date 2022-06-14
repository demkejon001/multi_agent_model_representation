import argparse
import parser
from typing import Callable, Optional

from tommas.data import debug_dataset, iterative_action_dataset, cooperative_gridworld_dataset


dataset_creation_data = dict()


def add_dataset(argparse_name: str, create_func: Callable, create_func_kwargs: Optional[dict] = None,
                verbose_name: Optional[str] = None):
    if argparse_name in dataset_creation_data:
        raise ValueError(f"argparse_name {argparse_name} has already been added. Please choose another name")
    if create_func_kwargs is None:
        create_func_kwargs = dict()
    dataset_creation_data[argparse_name] = (create_func, create_func_kwargs, verbose_name)


add_dataset(argparse_name="debug", create_func=debug_dataset.create_dataset,
            create_func_kwargs={"single_agent": False}, verbose_name="Debug")

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

