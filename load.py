import argparse

from experiments.experiment_base import download_wandb_modeller_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_link", type=str)
    args = parser.parse_args()
    download_wandb_modeller_artifact(args.artifact_link, 'data/models/')


if __name__=="__main__":
    main()
