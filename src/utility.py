from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
from typing import Any, Dict, Tuple

import itertools
import json
import os
import wandb


class TorchTensorboardLogger():
    def __init__(self, logdir: str):
        self.writer = TorchSummaryWriter(f"{logdir}/tensorboard")
        layout = {
            "stats": {
                "loss": ["Multiline", ["Loss/train", "Loss/valid"]],
            },
        }
        self.writer.add_custom_scalars(layout)

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class TorchWandbLogger():
    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        wandb.log({tag: scalar_value}, step)


def get_data() -> Tuple[Dataset, Dataset, Dataset]:
    train_val_data = datasets.MNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    # train/validation splitting
    # Note: transforms.ToTensor() scales the data range from [0; 255] to [0; 1]
    train_data, val_data = random_split(
        train_val_data,
        (int(0.8 * len(train_val_data)), int(0.2 * len(train_val_data)))
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        transform=transforms.ToTensor()
    )
    return train_data, val_data, test_data


def get_dataloaders(
    train_data: datasets.MNIST,
    val_data: datasets.MNIST,
    test_data: datasets.MNIST,
    batch_size: int = 512,
    num_workers: int = 1,
) -> Dict[str, DataLoader]:
    loaders = {
        'train' : DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        ),
        'val': DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        ),
        'test'  : DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        ),
    }
    return loaders


def get_outdir(exp_name: str, NeuralNet: str, exp_list: Dict, it: int):
    os.makedirs(exp_name, exist_ok=True)
    if exp_list is not None:
        trial_name = NeuralNet
        for k, v in exp_list.items():
            trial_name += f"_{k}_{str(v)}"
        folder_path = os.path.join(exp_name, f"{exp_name}_{trial_name}_it_{it}")
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    return exp_name


def get_hypp(config: Dict):
    """
    this function samples all parameters from the parameter sweep
    configuration and returns a list of dictionaries with each combination

    * input
    ** config: Dict - dictionary containing the experiment configuration
    * inner variables
    ** hyp_list: List[Tuple] - list of tuples made of all possible parameter combination
    * output
    ** exp_list: List[Dict] - list of dictionaries {parameter: value} for each combination
    """
    hyp_list = list(itertools.product(*config["parameters"].values()))
    print(hyp_list)
    exp_list = []
    for t in hyp_list:
        hyp_dict = {}
        for idx, name in enumerate(config["parameters"].keys()):
            hyp_dict[name] = t[idx]
        exp_list.append(hyp_dict)
    return exp_list


def load_config(config_name: str):
    # load the sweep config parameters
    jf_name = config_name if config_name[-5:] == ".json" else f"{config_name}.json"
    with open(jf_name) as jf:
        config_dict = json.load(jf)
    return config_dict
