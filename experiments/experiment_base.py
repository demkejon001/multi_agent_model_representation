import os
import wandb
from argparse import ArgumentParser, Namespace
import torch
import math

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.agent_modellers.tomnet_base import AgentModeller
from src.agent_modellers.model_factory import MODELS


modeller_extension = ".ckpt"


def add_experiment_args(parser: ArgumentParser):
    parser.add_argument('-d', '--description', type=str, required=True,
                        help='describe your experiment')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--num_minibatches', type=int, default=1000,
                        help='number of minibatches to train for (default: 40000)')
    parser.add_argument('--log_train_interval', type=int, default=100,
                        help='how many minibatches to wait before logging training metrics (default: 100)')
    parser.add_argument('--log_eval_interval', type=int, default=500,
                        help='how many minibatches to wait before logging evaluation metrics (default: 500)')
    parser.add_argument('--data_reload_interval', type=int, default=0,
                        help="how many minibatches to wait before reloading a new partition of a dataset (default: 0, i.e. doesn't reload)")
    parser.add_argument('--save_interval', type=int, default=500,
                        help='how many minibatches to wait before saving a model checkpoint (default: 500)')
    parser.add_argument('--precision', type=int, default=32,
                        help="Double precision (64), full precision (32) or half precision (16) default(: 32)")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--early_stopping', action="store_true", default=False,
                        help="use pytorch lightning's EarlyStopping callback")
    parser.add_argument('--model_checkpointing', action="store_true", default=False,
                        help="use pytorch lightning's ModelCheckpoint callback")
    parser.add_argument('--logging', action="store_true", default=False,
                        help="logs results onto WandB using pytorch lightning's WandBLogger")
    parser.add_argument('--log_model', action="store_true", default=False,
                        help="logs the trained model as a WandB artifact")


def add_dataset_args(parser: ArgumentParser):
    parser.add_argument("--remove_actions", action="store_true", default=False,
                        help="removes all actions from a trajectory")
    parser.add_argument("--remove_other_agents", action="store_true", default=False,
                        help="removes all agents not being modelled and their actions from a trajectory")


def extract_dataset_kwargs_from_args(args: Namespace):
    dataset_kwargs = dict()
    dataset_kwargs["remove_actions"] = args.remove_actions
    dataset_kwargs["remove_other_agents"] = args.remove_other_agents
    dataset_kwargs["transformer_get"] = True if "transformer" in args.model_type else False
    return dataset_kwargs


def extract_reload_dataloaders_every_n_epochs_from_args(args: Namespace):
    return math.ceil(args.data_reload_interval / args.log_eval_interval)


class ValidationCheckProgressBar(ProgressBar):
    def __init__(self, val_check_interval, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)
        self.val_check_interval = val_check_interval
        self._num_val_checks = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if (batch_idx + 1) % self.val_check_interval == 0:
            print('')
            self._num_val_checks += 1
            self.on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f"Val Check {self._num_val_checks}")

    @property
    def total_train_batches(self) -> int:
        return self.val_check_interval


def get_pl_trainer(args: Namespace, project_name, modeller_dirpath, modeller_filename, **kwargs):
    callbacks = [ValidationCheckProgressBar(args.log_eval_interval)]
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='overall_eval_loss', patience=10, mode='min', strict=False))
    if args.model_checkpointing:
        callbacks.append(ModelCheckpoint(monitor='overall_eval_loss', save_top_k=1, mode='min',
                                         dirpath=modeller_dirpath,
                                         filename=modeller_filename))
    wandb_logger = False
    if args.logging:
        wandb_logger = WandbLogger(project=project_name, job_type="train", save_dir=modeller_dirpath, config=vars(args),
                                   log_model=args.log_model, notes=args.description, **kwargs)

    trainer = pl.Trainer(logger=wandb_logger, checkpoint_callback=args.model_checkpointing, callbacks=callbacks,
                         gpus=1, check_val_every_n_epoch=1, max_steps=args.num_minibatches,
                         limit_train_batches=args.log_eval_interval, limit_val_batches=30,
                         log_every_n_steps=args.log_train_interval, precision=args.precision,
                         reload_dataloaders_every_n_epochs=extract_reload_dataloaders_every_n_epochs_from_args(args),
                         deterministic=args.deterministic, auto_lr_find=args.auto_lr_find,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         benchmark=True,
                         )
    return trainer


def download_wandb_modeller_artifact(artifact_link, save_dirpath):
    api = wandb.Api()
    artifact = api.artifact(artifact_link, type="model")
    artifact_dir = artifact.download()
    modeller_path = artifact_dir + '/model.ckpt'
    modeller = torch.load(modeller_path)
    model_type = modeller["hyper_parameters"]["model_type"]
    model_filename = modeller["hyper_parameters"]["model_name"]
    modeller = MODELS[model_type].load_from_checkpoint(modeller_path)
    print("Loading a", model_type, "model")
    if not os.path.exists(save_dirpath):
        os.makedirs(save_dirpath)
    print("Saving to", save_dirpath + model_filename + modeller_extension)
    torch.save(modeller, save_dirpath + model_filename + modeller_extension)


def load_modeller(modeller_path: str) -> AgentModeller:
    modeller = torch.load(modeller_path)
    if isinstance(modeller, dict):
        model_type = modeller["hyper_parameters"]["model_type"]
        modeller = MODELS[model_type].load_from_checkpoint(modeller_path)
    modeller.freeze()
    return modeller


def get_device(args: Namespace):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    return 'cuda' if use_cuda else 'cpu'
