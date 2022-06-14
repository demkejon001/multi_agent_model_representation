import argparse

from experiments.experiment_base import download_wandb_modeller_artifact

import os
import shutil
import wandb
import torch
from experiments.experiment_base import modeller_extension, MODELS


def main():
    artifact_links = [
        # lstm 64
        "model-37t40qgo:v0",
        "model-bjbxf1ex:v0",
        "model-2xurhtjr:v0",
        "model-39e1lful:v0",
        "model-1ctzfq7x:v0",
        # lstm 128
        "model-12pkia9t:v0",
        "model-3nyhfxl4:v0",
        "model-27nd1vyd:v0",
        "model-3eij82gv:v0",
        "model-2plea1l6:v0",
        # lstm 512
        "model-33go84rg:v0",
        "model-3nisopku:v0",
        "model-pm909nph:v0",
        "model-1gs5musf:v0",
        "model-2b8oljv1:v0",
        # ttx 64
        "model-3nu6b9vd:v0",
        "model-3i2zky8f:v0",
        "model-321e4q5w:v0",
        "model-3dsyw02u:v0",
        "model-f8toah4w:v0",
        # ttx 128
        "model-1kdxr60w:v0",
        "model-mvh80wmh:v0",
        "model-1qrj4yh0:v0",
        "model-29bzxj2k:v0",
        "model-36lgrgyi:v0",
        # ttx 512
        "model-1ry0ebjv:v0",
        "model-3h3sgf3m:v0",
        "model-30m6q1mz:v0",
        "model-19zjrzck:v0",
        "model-1207rmxh:v0",
    ]

    save_dirpath = 'data/models/gridworld/'
    all_model_params_ext = dict()

    if os.path.exists(save_dirpath):
        shutil.rmtree(save_dirpath)
    os.makedirs(save_dirpath)

    api = wandb.Api()
    for artifact_link in artifact_links:
        artifact = api.artifact('jpd0057/src/' + artifact_link, type="model")
        artifact_dir = artifact.download()
        modeller_path = artifact_dir + '/model.ckpt'
        modeller = torch.load(modeller_path)
        params = modeller["hyper_parameters"]

        char_size = params["char_embedding_size"]
        char_layer = params["char_n_layer"]
        char_head = params["char_n_head"]
        mental_size = params["mental_embedding_size"]
        mental_layer = params["mental_n_layer"]
        mental_head = params["mental_n_head"]
        if params["lstm_char"]:
            model_params_ext = f"lstm[{char_size},{char_layer}]_lstm[{mental_size},{mental_layer}]"
        else:
            if char_size >= 64:
                model_params_ext = f"ttx[{char_size},8,8]_ttx[{mental_size},{mental_layer},{mental_head}]"
            else:
                model_params_ext = f"ttx[{char_size},{char_layer},{char_head}]_ttx[{mental_size},{mental_layer},{mental_head}]"

        if model_params_ext not in all_model_params_ext:
            all_model_params_ext[model_params_ext] = 0
        if all_model_params_ext[model_params_ext] >= 6:
            print(f"{model_params_ext} has {all_model_params_ext[model_params_ext]} entries")
        all_model_params_ext[model_params_ext] += 1

        model_type = modeller["hyper_parameters"]["model_type"]
        model_filename = modeller["hyper_parameters"]["model_name"]

        seed_name = f"seed{all_model_params_ext[model_params_ext]}"
        model_filepath = f"{save_dirpath}{model_filename}_{model_params_ext}_{seed_name}{modeller_extension}"

        modeller = MODELS[model_type].load_from_checkpoint(modeller_path)
        print("Saving to", model_filepath)
        torch.save(modeller, model_filepath)


if __name__ == "__main__":
    main()
