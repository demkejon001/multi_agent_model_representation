import os
import pickle
import glob
from typing import List
import numpy as np
import random
import torch
import shutil


dataset_extension = '.pickle'


def save_dataset(dataset, save_dirpath, filename):
    if not os.path.exists(save_dirpath):
        os.makedirs(save_dirpath)
    file = open(save_dirpath + filename + dataset_extension, "wb")
    pickle.dump(dataset, file)
    file.close()


def delete_dirpath(dirpath):
    if os.path.exists(dirpath):
        try:
            shutil.rmtree(dirpath)
        except OSError as e:
            print(f"delete_dirpath error: {dirpath} : {e.strerror}")


def load_dataset(dataset_path):
    file = open(dataset_path, "rb")
    dataset = pickle.load(file)
    file.close()
    return dataset


def get_dataset_filepaths(dataset_dirpath: str, filename: str) -> List[str]:
    return glob.glob(dataset_dirpath + filename + "[0-9]*" + dataset_extension)


def seed_everything(seed: int):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
