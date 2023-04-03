import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import u_net
import utils

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path.cwd() / "TrainingData"
GEN_DIR = Path.cwd() / "GeneratedData"
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "segmentation_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 100
#0,01  0,001 0,0001 0,00001
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
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load training data and create DataLoader with batching and shuffling
dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
dataset_generated = utils.GeneratedDataset(partition["train"])
train_dataset = torch.utils.data.ConcatDataset([dataset,dataset_generated])

dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser, and loss function
loss_function = utils.DiceBCELoss()# TODO 
unet_model = u_net.UNet()# TODO 
optimizer = torch.optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)# TODO 
#optimizer = torch.optim.SGD(unet_model.parameters(), lr = LEARNING_RATE, momentum=0.99, nesterov = True)
#optimizer = torch.optim.ASGD(unet_model.parameters(), lr = LEARNING_RATE)
#optimizer = torch.optim.RMSprop(unet_model.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)

#optimizer = torch.optim.Adadelta(unet_model.parameters(), lr=LEARNING_RATE, rho=0.9, eps=1e-06)
#optimizer = torch.optim.Adagrad(unet_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)

#optimizer = torch.optim.NAdam(unet_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)

torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1, 
    patience=5, 
    verbose=False)


minimum_valid_loss = 10  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0

    # training iterations
    # tqdm is for timing iteratiions
    for inputs, labels in tqdm(dataloader, position=0):
        # needed to zero gradients in each iterations
        optimizer.zero_grad()
        outputs = unet_model(inputs)  # forward pass
        loss = loss_function(outputs, labels.float())
        loss.backward()  # backpropagate loss
        current_train_loss += loss.item()
        optimizer.step()  # update weights

    # evaluate validation loss
    with torch.no_grad():
        unet_model.eval()  # turn off training option for evaluation
        for inputs, labels in tqdm(valid_dataloader, position=0):
            outputs = unet_model(inputs)  # forward pass
            loss = loss_function(outputs, labels.float())
            current_valid_loss += loss.item()

        unet_model.train()  # turn training back on

    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
    )

    # if validation loss is improving, save model checkpoint
    # only start saving after 10 epochs
    if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss + TOLERANCE:
        minimum_valid_loss = current_valid_loss / len(valid_dataloader)
        if epoch > 9:
            torch.save(
                unet_model.cpu().state_dict(),
                CHECKPOINTS_DIR / f"u_net_{epoch}.pth",
            )

