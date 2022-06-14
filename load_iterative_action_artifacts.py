import argparse

from experiments.experiment_base import download_wandb_modeller_artifact

import os
import shutil
import wandb
import torch
from experiments.experiment_base import modeller_extension, MODELS


def main():
    artifact_links = [
        # lstm 64,1
        "model-g06w9ox5:v0",
        "model-2txuiual:v0",
        "model-3di42zwb:v0",
        "model-2e1xxld7:v0",
        "model-1xtry5pg:v0",
        # lstm 128,1
        "model-1dxy2yv1:v0",
        "model-1l1ub51z:v0",
        "model-17b8bmzp:v0",
        "model-zoclschp:v0",
        "model-2pi45cut:v0",
        # lstm 512,1
        "model-1zh2bglu:v0",
        "model-17tiy5f1:v0",
        "model-2xn378km:v0",
        "model-2e8hvzrd:v0",
        "model-1jkvxzlo:v0",
        # lstm 64,4,4
        "model-1awrx960:v0",
        "model-2enzrstl:v0",
        "model-312374qo:v0",
        "model-35tvq7nn:v0",
        "model-22dmjsd1:v0",
        # lstm 128,4,4
        "model-3lvoqk7e:v0",
        "model-gt4zratg:v0",
        "model-127kxvte:v0",
        "model-2l0i3hle:v0",
        "model-1hp0rzx0:v0",
        # lstm 512,4,4
        "model-366li5zw:v0",
        "model-2csfqfy1:v0",
        "model-bk4guut9:v0",
        "model-2fi0bmb7:v0",
        "model-ie1i53mm:v0",
    ]

    save_dirpath = 'data/models/iterative_action/'
    all_model_params_ext = dict()

    if os.path.exists(save_dirpath):
        shutil.rmtree(save_dirpath)
    os.makedirs(save_dirpath)

    api = wandb.Api()
    for artifact_link in artifact_links:
        artifact = api.artifact('jpd0057/mam_representation/' + artifact_link, type="model")
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


if __name__=="__main__":
    main()
