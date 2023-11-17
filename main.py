import argparse
import matplotlib.pyplot as plt
import os
import torch
import wandb

from datetime import datetime
from src.utility import (
    get_data,
    get_dataloaders,
    get_outdir,
    get_hypp,
    load_config,
    TorchTensorboardLogger,
    TorchWandbLogger
)
from src.models import CNN, MLP
from sklearn.metrics import accuracy_score
from typing import Dict, Optional, Union


def train_model(tb_logger, loaders, model, loss_func, n_epochs, lr, device):

    criterion = loss_func
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("start training")
    total_batches = 0
    for epoch in range(n_epochs):
        # train
        for batch, batch_data in enumerate(loaders["train"]):
            # Backprogation
            optimizer.zero_grad()
            x, y_target = batch_data
            x = x.to(device)
            y_target = y_target.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y_target)
            loss.backward()
            optimizer.step()
            total_batches += 1

            tb_logger.log_scalar("Loss/train", loss.item(), total_batches)

        val_loss = 0
        valid_batch = 0
        accuracy = 0
        # validation
        with torch.no_grad():
            model.eval()
            for valid_batch, batch_data in enumerate(loaders["val"]):
                x, y_target = batch_data
                x = x.to(device)  # type: ignore
                y_target = y_target.to(device)
                # model evaluation
                y_pred = model(x)
                loss = criterion(y_pred, y_target)
                val_loss += loss.item()
                accuracy += accuracy_score(torch.max(y_pred, 1)[1], y_target)

        tb_logger.log_scalar("Loss/valid", val_loss / (valid_batch + 1), total_batches)
        tb_logger.log_scalar("Acc/valid", accuracy / (valid_batch + 1), total_batches)
        model.train()

        # Learning rate
        lrs = []
        for grp in optimizer.param_groups:
            lrs.append(grp["lr"])
        tb_logger.log_scalar("lr", lrs[0], total_batches)

        print(
            f"epoch {epoch + 1}/{n_epochs}, \n"
            f"train loss {loss.item()}, val loss {val_loss / (valid_batch + 1)}, "
            f"val accuracy {accuracy / (valid_batch + 1)}"
        )

    print("finished training")
    return accuracy / (valid_batch + 1), val_loss / (valid_batch + 1)


def test_model(test_dataloader, model, loss_func):
    # Test the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        correct = 0
        for batch, batch_data in enumerate(test_dataloader):
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)
            y_pred = model(images)
            test_loss += loss_func(y_pred, labels).item()
            # torch.max(tensor, X) returns a tuple (max_value, max_index)
            # for each array along the X dim of tensor
            correct += (torch.max(y_pred, 1)[1] == labels).sum().item()
    print(f"test loss: {test_loss/batch}, test accuracy: {correct}/{len(test_dataloader.dataset)} "
          f"i.e. {correct/len(test_dataloader.dataset)}")


def plot_inference(out_dir, test_dataset, model):
    figure = plt.figure(figsize=(20, 16))
    cols, rows = 10, 10
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_data), size=(1,)).item()
        img = test_data[sample_idx][0]
        pred_label = torch.max(model(torch.unsqueeze(test_data[sample_idx][0], dim=0)), 1)[1].item()
        figure.add_subplot(rows, cols, i)
        if test_data[sample_idx][1] != pred_label:
            plt.title(f"{test_data[sample_idx][1]} vs {pred_label}", x=0.5, y=0.95, color="red")
        else:
            plt.title(f"{test_data[sample_idx][1]} vs {pred_label}", x=0.5, y=0.95, color="black")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.savefig(out_dir + "/test_sample.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--batch-size", type=int, default=512, help="batch size for train loop."
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="name of the experiment config file."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train on."
    )
    parser.add_argument(
        "--json", action="store_true", help="use a json config file."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate."
    )
    parser.add_argument(
        "--model", type=str, default="CNN", help="model to train/execute: CNN or MLP."
    )
    parser.add_argument("--test", action="store_true", help="to test.")
    parser.add_argument("--train", action="store_true", help="to train.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=datetime.now().strftime("experiment-%Y%m%dT%H%M%S"),
        help="where output data are written.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="torch manual seed."
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use a wandb config file."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start study with device: {device}")

    # non-variable training settings
    torch.manual_seed(args.seed)
    train_data, val_data, test_data = get_data()
    loss_func = torch.nn.CrossEntropyLoss()

    # logger
    logger: Optional[Union[TorchTensorboardLogger, TorchWandbLogger]] = None

    # configured execution with multiple trials
    if args.json:
        # load parameter sweep config
        config_dict = load_config(args.config)

        n_trial = config_dict["n_trial"]

        # NeuralNet architecture sweep
        for NeuralNet in config_dict["models"].keys():
            hyp_dict_list = get_hypp(config_dict["models"][NeuralNet])

            # parameter sweep training
            for sweep in hyp_dict_list:
                for it in range(n_trial):
                    print(f"\ntrial: {it + 1}/{n_trial}")
                    out_dir = get_outdir(config_dict["name"], NeuralNet, sweep, it)
                    logger = TorchTensorboardLogger(logdir=out_dir)
                    if NeuralNet == "CNN":
                        model: torch.nn.Module = CNN()
                    elif NeuralNet == "MLP":
                        model = MLP()
                    else:
                        raise Exception("Model not implemented")
                    loaders: Dict[str, torch.utils.data.DataLoader] = get_dataloaders(
                        train_data, val_data, test_data, sweep["batch_size"]
                    )
                    n_epochs = sweep["epochs"]
                    lr = sweep["lr"]
                    print(f"launch training for architecture: {NeuralNet} and parameters: {sweep}")
                    train_model(logger, loaders, model, loss_func, n_epochs, lr, device)
                    test_model(loaders["test"], model, loss_func)
                    # delete model to enforce weights resetting
                    del model

    # configured execution with wandb
    elif args.wandb:
        config_dict = load_config(args.config)

        def wandb_run():
            wandb.init()
            logger = TorchWandbLogger()
            if NeuralNet == "CNN":
                model = CNN()
            elif NeuralNet == "MLP":
                model = MLP()
            else:
                raise Exception("Model not implemented")
            loaders = get_dataloaders(
                train_data, val_data, test_data, wandb.config["batch_size"]
            )
            n_epochs = wandb.config["epochs"]
            lr = wandb.config["lr"]
            val_acc, val_loss = train_model(logger, loaders, model, loss_func, n_epochs, lr, device)
            wandb.log({sweep_configuration["metric"]["name"]: val_acc})
            # delete model to enforce weights resetting
            del model

        for NeuralNet in config_dict["models"].keys():
            print(f"launch wandb sweep for architecture: {NeuralNet}")
            sweep_configuration = config_dict["models"][NeuralNet]
            sweep_id = wandb.sweep(sweep=sweep_configuration, project=config_dict["name"])
            wandb.agent(sweep_id, function=wandb_run)

    # unique execution
    else:
        print(args)

        if args.model == "CNN":
            model = CNN()
        elif args.model == "MLP":
            model = MLP()
        else:
            raise Exception("Model not implemented")

        os.makedirs(args.out_dir, exist_ok=True)
        logger = TorchTensorboardLogger(logdir=args.out_dir)

        loaders = get_dataloaders(
            train_data, val_data, test_data, args.batch_size
        )

        if args.train:
            train_model(logger, loaders, model, loss_func, args.epochs, args.lr, device)
        if args.test:
            test_model(loaders["test"], model, loss_func)
            plot_inference(args.out_dir, test_data, model)
