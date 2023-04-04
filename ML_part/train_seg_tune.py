# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:49:03 2023

@author: raque
"""

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

import u_net
import utils

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial



# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path.cwd() / "TrainingData"
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "segmentation_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 1
LEARNING_RATE = 1e-4
TOLERANCE = 0.01  # for early stopping

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
# partition = {
#     "train": patients[:-NO_VALIDATION_PATIENTS],
#     "validation": patients[-NO_VALIDATION_PATIENTS:],
# }

# # load training data and create DataLoader with batching and shuffling
# dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
# dataloader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     pin_memory=True,
# )

# # load validation data
# valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
# valid_dataloader = DataLoader(
#     valid_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     pin_memory=True,
# )

# initialise model, optimiser, and loss function
loss_function = utils.DiceBCELoss()# TODO 
unet_model = u_net.UNet()# TODO 
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)# TODO 

# torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 
#     mode='min', 
#     factor=0.1, 
#     patience=5, 
#     verbose=False)


minimum_valid_loss = 10  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary


def train_net(config, checkpoint_dir=None, data_dir=None):
    net = unet_model(config["l1"], config["l2"])
    #net = unet_model()


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)


    loss_function = utils.DiceBCELoss()# TODO 
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    patients = [
        path
        for path in DATA_DIR.glob("*")
        if not any(part.startswith(".") for part in path.parts)
    ]
    random.shuffle(patients)

    # split in training/validation after shuffling
    partition = {
        "train": patients[:-NO_VALIDATION_PATIENTS],
        "validation": patients[-NO_VALIDATION_PATIENTS:],
    }

    # load training data and create DataLoader with batching and shuffling
    dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    # load validation data
    valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                #print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,running_loss / epoch_steps))
                print("[%d, %5d] loss: %.3f" % (epoch, i, running_loss / epoch_steps))

                running_loss = 0.0
                

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valid_dataloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = loss_function
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")



def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    # directorys with data and to store training checkpoints and logs
    data_dir = Path.cwd() / "TrainingData"
    CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights"
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # config = {
    #     "lr": tune.loguniform(1e-4, 1e-1),
    # }
    
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
    }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='min', 
    #     factor=0.1, 
    #     patience=5, 
    #     verbose=False)
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss"])
    
    result = tune.run(
        partial(train_net, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    # best_trained_model = unet_model(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
    
    
# # training loop
# for epoch in range(N_EPOCHS):
#     current_train_loss = 0.0
#     current_valid_loss = 0.0
    
#     # TODO 
#     # training iterations


#     # evaluate validation loss
#     with torch.no_grad():
#         unet_model.eval()
#         # TODO 

#         unet_model.train()

#     # write to tensorboard log
#     writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
#     writer.add_scalar(
#         "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
#     )

#     # if validation loss is improving, save model checkpoint
#     # only start saving after 10 epochs
#     if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss + TOLERANCE:
#         minimum_valid_loss = current_valid_loss / len(valid_dataloader)
#         if epoch > 9:
#             torch.save(
#                 unet_model.cpu().state_dict(),
#                 CHECKPOINTS_DIR / f"u_net_{epoch}.pth",
#             )
