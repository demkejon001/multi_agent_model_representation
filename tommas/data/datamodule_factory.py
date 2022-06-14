from typing import List, Tuple
import numpy as np
from argparse import Namespace, ArgumentParser

from pytorch_lightning import LightningDataModule

from tommas.data.gridworld_transforms import GridworldTrajectoryTransform, FullTrajectory, PastCurrentStateSplit, \
    IterativeActionFullTrajectory, IterativeActionPastCurrentSplit, TokenIDTrajectory, \
    IterativeActionFullPastCurrentSplit, CoopGridworldTrajectory
from tommas.data.agent_trajectory_fetcher import AgentTrajectoryFetcher, AgentGridworldFilepaths, \
    IterativeActionTrajectoryFilepaths
from tommas.data.gridworld_dataset import GridworldDataset
from tommas.data.gridworld_datamodule import GridworldDataModule
# Debug
from tommas.data.debug_dataset import get_dataset_filepaths as get_debug_dataset_filepaths
# Action IA/JA
from tommas.data.iterative_action_dataset import get_dataset_filepaths as get_iterative_action_dataset_filepaths
from tommas.data.iterative_action_fake_ja_dataset import get_dataset_filepaths as get_iterative_action_fake_ja_dataset_filepaths
# Random Policy
from tommas.data.random_policy_dataset import get_dataset_filepaths as get_random_policy_dataset_filepaths
# Gridworld
from tommas.data.cooperative_gridworld_dataset import get_dataset_filepaths as get_gridworld_dataset_filepaths
from tommas.data.iterative_action_gridworld_dataset import get_dataset_filepaths as get_iter_action_gridworld_dataset_filepaths
from tommas.data.multi_strategy_gridworld_dataset import get_dataset_filepaths as get_multi_strat_gridworld_dataset_filepaths


DATASETS = {
    "debug":  get_debug_dataset_filepaths,
    "action_ia2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": True}),
    "action_ja2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": False}),
    "action_ia3": (get_iterative_action_dataset_filepaths, {"num_opponents": 2, "is_ia": True}),
    "action_ja3": (get_iterative_action_dataset_filepaths, {"num_opponents": 2, "is_ia": False}),
    "action_ia4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": True}),
    "action_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False}),
    "action_ia5": (get_iterative_action_dataset_filepaths, {"num_opponents": 4, "is_ia": True}),
    "action_ja5": (get_iterative_action_dataset_filepaths, {"num_opponents": 4, "is_ia": False}),
    "action_ia6": (get_iterative_action_dataset_filepaths, {"num_opponents": 5, "is_ia": True}),
    "action_ja6": (get_iterative_action_dataset_filepaths, {"num_opponents": 5, "is_ia": False}),
    "action_ia7": (get_iterative_action_dataset_filepaths, {"num_opponents": 6, "is_ia": True}),
    "action_ja7": (get_iterative_action_dataset_filepaths, {"num_opponents": 6, "is_ia": False}),
    "action_ia8": (get_iterative_action_dataset_filepaths, {"num_opponents": 7, "is_ia": True}),
    "action_ja8": (get_iterative_action_dataset_filepaths, {"num_opponents": 7, "is_ia": False}),
    "action_pattern_ia2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": True, "tags": "pattern"}),
    "action_pattern_ja2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": False, "tags": "pattern"}),
    "action_simple_ia2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": True, "tags": "simple"}),
    "action_simple_ia4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": True, "tags": "simple"}),
    "action_simple_ia8": (get_iterative_action_dataset_filepaths, {"num_opponents": 7, "is_ia": True, "tags": "simple"}),
    "action_mixed_trigger_ja2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": False, "tags": "mixed_trigger"}),
    "action_mixed_trigger_ja3": (get_iterative_action_dataset_filepaths, {"num_opponents": 2, "is_ia": False, "tags": "mixed_trigger"}),
    "action_mixed_trigger_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "mixed_trigger"}),
    "action_mixed_trigger_ja5": (get_iterative_action_dataset_filepaths, {"num_opponents": 4, "is_ia": False, "tags": "mixed_trigger"}),
    "action_mixed_trigger_ja6": (get_iterative_action_dataset_filepaths, {"num_opponents": 5, "is_ia": False, "tags": "mixed_trigger"}),
    "action_mixed_trigger_ja7": (get_iterative_action_dataset_filepaths, {"num_opponents": 6, "is_ia": False, "tags": "mixed_trigger"}),
    "action_mixed_trigger_ja8": (get_iterative_action_dataset_filepaths, {"num_opponents": 7, "is_ia": False, "tags": "mixed_trigger"}),
    "action_fake_mixed_trigger_ja2": (get_iterative_action_fake_ja_dataset_filepaths, {"num_opponents": 1, "is_ia": False, "tags": "fake_mixed_trigger"}),
    "action_fake_mixed_trigger_ja7": (get_iterative_action_fake_ja_dataset_filepaths, {"num_opponents": 6, "is_ia": False, "tags": "fake_mixed_trigger"}),
    "action_multi_strat_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "multi_strat"}),
    "action_mirror_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "mirror"}),
    "action_action_pattern_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "action_pattern"}),
    "action_wsls_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "wsls"}),
    "action_grim_trigger_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "grim_trigger"}),
    "action_mixed_trigger_pattern_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "mixed_trigger_pattern"}),
    "action_mirror_ja2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": False, "tags": "mirror"}),
    "action_action_pattern_ja2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": False, "tags": "action_pattern"}),
    "action_wsls_ja2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": False, "tags": "wsls"}),
    "action_grim_trigger_ja2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": False, "tags": "grim_trigger"}),
    "action_mixed_trigger_pattern_ja2": (get_iterative_action_dataset_filepaths, {"num_opponents": 1, "is_ia": False, "tags": "mixed_trigger_pattern"}),
    "random_policy": get_random_policy_dataset_filepaths,
    "random_policy_0.01": (get_random_policy_dataset_filepaths, {"alpha": 0.01}),
    "random_policy_0.03": (get_random_policy_dataset_filepaths, {"alpha": 0.03}),
    "random_policy_0.1": (get_random_policy_dataset_filepaths, {"alpha": 0.1}),
    "random_policy_0.3": (get_random_policy_dataset_filepaths, {"alpha": 0.3}),
    "random_policy_1.0": (get_random_policy_dataset_filepaths, {"alpha": 1.0}),
    "random_policy_3.0": (get_random_policy_dataset_filepaths, {"alpha": 3.0}),
    "random_policy_single": (get_random_policy_dataset_filepaths, {"is_single_agent": True}),
    "random_policy_single_0.01": (get_random_policy_dataset_filepaths, {"alpha": 0.01, "is_single_agent": True}),
    "random_policy_single_0.03": (get_random_policy_dataset_filepaths, {"alpha": 0.03, "is_single_agent": True}),
    "random_policy_single_0.1": (get_random_policy_dataset_filepaths, {"alpha": 0.1, "is_single_agent": True}),
    "random_policy_single_0.3": (get_random_policy_dataset_filepaths, {"alpha": 0.3, "is_single_agent": True}),
    "random_policy_single_1.0": (get_random_policy_dataset_filepaths, {"alpha": 1.0, "is_single_agent": True}),
    "random_policy_single_3.0": (get_random_policy_dataset_filepaths, {"alpha": 3.0, "is_single_agent": True}),
    "gridworld": get_gridworld_dataset_filepaths,
    "iterative_action_gridworld": get_iter_action_gridworld_dataset_filepaths,
    "multi_strategy_gridworld": get_multi_strat_gridworld_dataset_filepaths,
}


def get_datasets_abbreviation_str(args: Namespace):
    dataset_abbreviations = {
        "debug": "Debug",
        "action_ia2": "IAIter2",
        "action_ja2": "JAIter2",
        "action_ia3": "IAIter3",
        "action_ja3": "JAIter3",
        "action_ia4": "IAIter4",
        "action_ja4": "JAIter4",
        "action_ia5": "IAIter5",
        "action_ja5": "JAIter5",
        "action_ia6": "IAIter6",
        "action_ja6": "JAIter6",
        "action_ia7": "IAIter7",
        "action_ja7": "JAIter7",
        "action_ia8": "IAIter8",
        "action_ja8": "JAIter8",
        "action_pattern_ia2": "IAIterPtrn2",
        "action_pattern_ja2": "JAIterPtrn2",
        "action_simple_ia2": "IAIterSimpl2",
        "action_simple_ia4": "IAIterSimpl4",
        "action_simple_ia8": "IAIterSimpl8",
        "action_mixed_trigger_ja2": "JAIterMixTgr2",
        "action_mixed_trigger_ja3": "JAIterMixTgr3",
        "action_mixed_trigger_ja4": "JAIterMixTgr4",
        "action_mixed_trigger_ja5": "JAIterMixTgr5",
        "action_mixed_trigger_ja6": "JAIterMixTgr6",
        "action_mixed_trigger_ja7": "JAIterMixTgr7",
        "action_mixed_trigger_ja8": "JAIterMixTgr8",
        "action_fake_mixed_trigger_ja2": "JAIterFkMixTgr2",
        "action_fake_mixed_trigger_ja7": "JAIterFkMixTgr7",
        "action_multi_strat_ja4": "JAMultiStrat4",
        "action_mirror_ja4": "JAMirror4",
        "action_action_pattern_ja4": "JAPtrn4",
        "action_wsls_ja4": "JAWSLS4",
        "action_grim_trigger_ja4": "JAGrim4",
        "action_mixed_trigger_pattern_ja4": "JAMixTgrPtrn4",
        "action_mirror_ja2": "JAMirror2",
        "action_action_pattern_ja2": "JAPtrn2",
        "action_wsls_ja2": "JAWSLS2",
        "action_grim_trigger_ja2": "JAGrim2",
        "action_mixed_trigger_pattern_ja2": "JAMixTgrPtrn2",
        "random_policy": "RandPolicy",
        "random_policy_0.01": "RandPolicy0.01",
        "random_policy_0.03": "RandPolicy0.03",
        "random_policy_0.1": "RandPolicy0.1",
        "random_policy_0.3": "RandPolicy0.3",
        "random_policy_1.0": "RandPolicy1.0",
        "random_policy_3.0": "RandPolicy3.0",
        "random_policy_single": "RandPolicyS",
        "random_policy_single_0.01": "RandPolicyS0.01",
        "random_policy_single_0.03": "RandPolicyS0.03",
        "random_policy_single_0.1": "RandPolicyS0.1",
        "random_policy_single_0.3": "RandPolicyS0.3",
        "random_policy_single_1.0": "RandPolicyS1.0",
        "random_policy_single_3.0": "RandPolicyS3.0",
        "gridworld": "Grid",
        "iterative_action_gridworld": "IterGrid",
        "multi_strategy_gridworld": "MultiStratGrid",
    }
    return "[" + ",".join(map(dataset_abbreviations.get, args.dataset)) + "]"


def get_filepaths(args: Namespace):
    train_filepaths = []
    val_filepaths = []
    datasets = dataset_check(args)
    for dataset in datasets:
        if isinstance(DATASETS[dataset], tuple):
            dataset_fn, dataset_kwargs = DATASETS[dataset][0], DATASETS[dataset][1]
            train_filepaths.extend(dataset_fn(True, **dataset_kwargs))
            val_filepaths.extend(dataset_fn(False, **dataset_kwargs))
        else:
            train_filepaths.extend(DATASETS[dataset](True))
            val_filepaths.extend(DATASETS[dataset](False))
    return train_filepaths, val_filepaths


def get_agent_trajectory_fetcher(args: Namespace) -> Tuple[AgentTrajectoryFetcher, AgentTrajectoryFetcher]:
    datasets = dataset_check(args)
    train_filepaths, val_filepaths = get_filepaths(args)
    if "action" in datasets[0]:
        fetcher_cls = IterativeActionTrajectoryFilepaths
    else:
        fetcher_cls = AgentGridworldFilepaths
    return fetcher_cls(train_filepaths), fetcher_cls(val_filepaths)


def get_gridworld_transform_type(args: Namespace):
    datasets = dataset_check(args)
    model_type = args.model

    if "gridworld" in model_type:
        return "gridworld"

    if "past_current" in model_type:
        return "iterative_action_full_past_current_state_split"

    if "transformer" in model_type or "lstm" in model_type:
        if "action" in datasets[0]:
            transform_type = "iterative_action_full_trajectory"
        else:
            transform_type = "full_trajectory"
    else:
        if "action" in datasets[0]:
            transform_type = "iterative_action_past_current_split"
        else:
            transform_type = "past_current_state_split"

    if "huggingface" in model_type:
        transform_type = "token_id"

    return transform_type


def get_gridworld_transform(args: Namespace) -> GridworldTrajectoryTransform:
    transform_type = get_gridworld_transform_type(args)
    if transform_type == "gridworld":
        return CoopGridworldTrajectory()
    elif transform_type == "full_trajectory":
        # return FullTrajectory(remove_actions=args.remove_actions, remove_other_agents=args.remove_other_agents,
        #                       attach_agent_ids=args.attach_agent_ids, keep_goals=args.keep_goals)
        return FullTrajectory()
    elif transform_type == "past_current_state_split":
        # return PastCurrentStateSplit(remove_actions=args.remove_actions, concat_past_traj=args.concat_past_traj,
        #                              remove_other_agents=args.remove_other_agents,
        #                              attach_agent_ids=args.attach_agent_ids, keep_goals=args.keep_goals)
        return PastCurrentStateSplit()
    elif transform_type == "iterative_action_past_current_split":
        # return IterativeActionPastCurrentSplit(concat_past_traj=args.concat_past_traj)
        return IterativeActionPastCurrentSplit()
    elif transform_type == "iterative_action_full_trajectory":
        return IterativeActionFullTrajectory()
    elif transform_type == "token_id":
        return TokenIDTrajectory()
    elif transform_type == "iterative_action_full_past_current_state_split":
        return IterativeActionFullPastCurrentSplit()
    else:
        raise ValueError(f"transform_type {transform_type} is not recognized")


def make_datamodule(args: Namespace) -> LightningDataModule:
    train_fetcher, test_fetcher = get_agent_trajectory_fetcher(args)
    trajectory_transform = get_gridworld_transform(args)
    collate_fn = trajectory_transform.get_collate_fn()

    train_dataset = GridworldDataset(train_fetcher, trajectory_transform)
    test_dataset = GridworldDataset(test_fetcher, trajectory_transform)

    return GridworldDataModule(train_dataset, test_dataset, collate_fn, batch_size=args.batch_size, pin_memory=True,
                               num_workers=args.num_workers, num_batches_in_epoch=args.log_eval_interval,
                               n_past=args.n_past)


def add_datamodule_specific_args(parser: ArgumentParser, args: Namespace) -> ArgumentParser:
    parser = GridworldDataModule.add_datamodule_specific_args(parser)
    return parser


def dataset_check(args: Namespace) -> List[str]:
    datasets = args.dataset
    if isinstance(datasets, str):
        datasets = [datasets]
    datasets = list(np.sort(datasets))
    args.dataset = datasets
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError("dataset %s, is not a known dataset. Choose from the following datasets: %s" %
                             (dataset, list(DATASETS.keys())))
    return datasets
