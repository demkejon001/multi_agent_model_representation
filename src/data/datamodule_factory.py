from typing import List, Tuple
import numpy as np
from argparse import Namespace, ArgumentParser

from pytorch_lightning import LightningDataModule

from src.data.gridworld_transforms import GridworldTrajectoryTransform, \
    IterativeActionFullPastCurrentSplit, CoopGridworldTrajectory
from src.data.agent_trajectory_fetcher import AgentTrajectoryFetcher, AgentGridworldFilepaths, \
    IterativeActionTrajectoryFilepaths
from src.data.gridworld_dataset import GridworldDataset
from src.data.gridworld_datamodule import GridworldDataModule
from src.data.iterative_action_dataset import get_dataset_filepaths as get_iterative_action_dataset_filepaths
from src.data.cooperative_gridworld_dataset import get_dataset_filepaths as get_gridworld_dataset_filepaths


DATASETS = {
    "action_mirror_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "mirror"}),
    "action_wsls_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "wsls"}),
    "action_grim_trigger_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "grim_trigger"}),
    "action_mixed_trigger_pattern_ja4": (get_iterative_action_dataset_filepaths, {"num_opponents": 3, "is_ia": False, "tags": "mixed_trigger_pattern"}),
    "gridworld": get_gridworld_dataset_filepaths,
}


def get_datasets_abbreviation_str(args: Namespace):
    dataset_abbreviations = {
        "action_multi_strat_ja4": "JAMultiStrat4",
        "action_mirror_ja4": "JAMirror4",
        "action_wsls_ja4": "JAWSLS4",
        "action_grim_trigger_ja4": "JAGrim4",
        "action_mixed_trigger_pattern_ja4": "JAMixTgrPtrn4",
        "gridworld": "Grid",
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

    return transform_type


def get_gridworld_transform(args: Namespace) -> GridworldTrajectoryTransform:
    transform_type = get_gridworld_transform_type(args)
    if transform_type == "gridworld":
        return CoopGridworldTrajectory()
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
