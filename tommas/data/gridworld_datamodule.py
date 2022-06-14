from argparse import ArgumentParser

from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning.core import LightningDataModule

from tommas.data.gridworld_dataset import GridworldDatasetBatchSampler


class GridworldDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, collate_fn,
                 batch_size=16, pin_memory=False, num_workers=0, empty_current_traj_probability=1/35, n_past=-1,
                 additional_dataset_kwargs=None, num_batches_in_epoch=None):
        super().__init__()
        self.train = train_dataset
        self.val = val_dataset
        self.collate_fn = collate_fn
        self.observation_space = self.train.observation_space
        self.action_space = self.train.action_space
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.empty_current_traj_probability = empty_current_traj_probability
        self.n_past = n_past
        if additional_dataset_kwargs is None:
            additional_dataset_kwargs = {}
        self.additional_dataset_kwargs = additional_dataset_kwargs
        self.num_batches_in_epoch = num_batches_in_epoch

    @staticmethod
    def add_datamodule_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("GridworldDataModule")
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--no_pin_memory', action="store_true", default=False)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--empty_current_traj_probability', type=float, default=1/35)
        parser.add_argument('--n_past', type=int, default=-1)
        return parent_parser

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def _get_dataloader(self, dataset):
        if self.num_batches_in_epoch is not None:
            sampler = RandomSampler(dataset, num_samples=self.num_batches_in_epoch * self.batch_size, replacement=True)
        else:
            sampler = RandomSampler(dataset)
        batch_sampler = GridworldDatasetBatchSampler(sampler=sampler, batch_size=self.batch_size, drop_last=True,
                                                     min_num_episodes=min(10, dataset.min_num_episodes_per_agent),
                                                     current_traj_len=-1,
                                                     empty_current_traj_probability=self.empty_current_traj_probability,
                                                     n_past=self.n_past)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=self.collate_fn,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)

    def train_dataloader(self):
        return self._get_dataloader(self.train)

    def val_dataloader(self):
        return self._get_dataloader(self.val)
