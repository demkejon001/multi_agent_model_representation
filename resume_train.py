import os
from argparse import ArgumentParser, Namespace
import numpy as np
import wandb

import torch
from pytorch_lightning import seed_everything

from experiments.experiment_base import get_pl_trainer
from tommas.agent_modellers.model_factory import add_model_specific_args, get_model_name, MODELS
from tommas.data.datamodule_factory import make_datamodule, add_datamodule_specific_args


project_name = 'tommas'


def add_trainer_args(parent_parser: ArgumentParser):
    parser = parent_parser.add_argument_group("pl.Trainer")
    parser.add_argument('--num_minibatches', type=int, default=1000,
                        help='number of minibatches to train for (default: 1000)')
    parser.add_argument('--log_train_interval', type=int, default=100,
                        help='how many minibatches to wait before logging training metrics (default: 100)')
    parser.add_argument('--log_eval_interval', type=int, default=500,
                        help='how many minibatches to wait before logging evaluation metrics (default: 500)')
    parser.add_argument('--data_reload_interval', type=int, default=0,
                        help="how many minibatches to wait before reloading a new partition of a dataset (default: 0, i.e. doesn't reload)")
    parser.add_argument('--save_interval', type=int, default=500,
                        help='how many minibatches to wait before saving a model checkpoint (default: 500)')
    parser.add_argument('--precision', type=int, default=32,
                        help="Double precision (64), full precision (32) or half precision (16) (default: 32)")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help="Accumulates grads every k batches or as set up in the dict. (default: 1)")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--seed', nargs='?', type=int, const=1,
                        help='random seed')
    parser.add_argument('--early_stopping', action="store_true", default=False,
                        help="use pytorch lightning's EarlyStopping callback")
    parser.add_argument('--model_checkpointing', action="store_true", default=False,
                        help="use pytorch lightning's ModelCheckpoint callback")
    parser.add_argument('--deterministic', action="store_true", default=False,
                        help="sets whether PyTorch operations must use deterministic algorithms (default: False)")
    parser.add_argument('--auto_lr_find', action="store_true", default=False,
                        help="If set to True, will make trainer.tune() run a learning rate finder (default: False)")
    parser.add_argument('--logging', action="store_true", default=False,
                        help="logs results onto WandB using pytorch lightning's WandBLogger")
    return parent_parser


def add_experiment_args(parser: ArgumentParser):
    parser.add_argument("--model", type=str, required=True,
                        help="the model type you wish to train")
    parser.add_argument("--dataset", nargs="+", type=str, required=True,
                        help="the dataset(s) you want to train on")
    parser.add_argument("-d", "--description", type=str, required=True,
                        help='describe your experiment')
    parser.add_argument("-l", "--artifact_link", type=str, required=True,
                        help='wandb model link')
    parser.add_argument('--log_model', action="store_true", default=False,
                        help="logs the trained model as a WandB artifact")
    return parser


def get_seed(args: Namespace) -> int:
    if args.seed is None:
        args.seed = np.random.randint(0, 1000000)
    return args.seed


def train(args: Namespace):
    get_seed(args)
    seed_everything(args.seed, True)
    datamodule = make_datamodule(args)

    artifact_dir = "artifacts/" + args.artifact_link.split("/")[-1]
    if not os.path.exists(artifact_dir):
        api = wandb.Api()
        artifact = api.artifact(args.artifact_link, type="model")
        artifact_dir = artifact.download()

    modeller_path = artifact_dir + '/model.ckpt'
    modeller = torch.load(modeller_path)
    model_type = modeller["hyper_parameters"]["model_type"]
    model = MODELS[model_type].load_from_checkpoint(modeller_path)

    model_dirpath = 'data/models/'
    model_filename = get_model_name(args)
    trainer = get_pl_trainer(args, project_name, model_dirpath, model_filename)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()

    # add PROGRAM level args
    parser = add_experiment_args(parser)
    temp_args, _ = parser.parse_known_args()

    # add model specific args
    parser = add_model_specific_args(parser, temp_args)
    # add dataset specific args
    parser = add_datamodule_specific_args(parser, temp_args)
    # add trainer specific args
    parser = add_trainer_args(parser)

    args = parser.parse_args()
    train(args)
