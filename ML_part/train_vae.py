import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from matplotlib.pyplot import imshow, figure

import utils
import vae

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path.cwd() / "TrainingData"
CHECKPOINTS_DIR = Path.cwd() / "vae_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "vae_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 200
DECAY_LR_AFTER = 50
LEARNING_RATE = 1e-4
DISPLAY_FREQ = 10

# dimension of VAE latent space
Z_DIM = 256


# function to reduce the
def lr_lambda(the_epoch):
    """Function for scheduling learning rate"""
    return (
        1.0
        if the_epoch < DECAY_LR_AFTER
        else 1 - float(the_epoch - DECAY_LR_AFTER) / (N_EPOCHS - DECAY_LR_AFTER)
    )


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
dataloader = DataLoader(
    dataset,
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

# initialise model, optimiser
vae_model = vae.VAE() # TODO 
optimizer = torch.optim.Adam(vae_model.parameters()) # TODO 
# add a learning rate scheduler based on the lr_lambda function
scheduler = lr_lambda(0) #torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # TODO

# training loop
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0
    
    # TODO 
    # training iterations
    vae_model.train()
    # TODO 
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader, position=0):

        # zero the parameter gradients
        optimizer.zero_grad()
        
        x_recon = vae_model(inputs) # forward
        # Evaluate loss
        mu = 1
        logvar = np.log(mu)
        loss = vae.vae_loss(inputs, x_recon, mu, logvar) # get loss
        
        # Backward pass
        loss.backward()# backpropaate loss
        
        current_train_loss += loss.item()
        optimizer.step() # Update parameters (optimize)
    
    scheduler.step() # Update learning rate

    # evaluate validation loss
    with torch.no_grad():
        vae_model.eval()
        for inputs, x_real in tqdm(valid_dataloader, position=0): # TODO 
            x_recon = vae_model(inputs) # forward pass
            loss = vae.vae_loss(inputs, x_recon, mu, logvar)
            current_valid_loss += loss.item()
            
        vae_model.train()
    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
    )
    scheduler.step() # step the learning step scheduler

    # save examples of real/fake images
    if (epoch + 1) % DISPLAY_FREQ == 0:
        img_grid = make_grid(
            torch.cat((x_recon[:5], x_real[:5])), nrow=5, padding=12, pad_value=-1
        )
        writer.add_image(
            "Real_fake", np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5, epoch + 1
        )
        
    # TODO: sample noise 
    num_preds = 16
    noise = vae.get_noise(num_preds, IMAGE_SIZE)
    
    # TODO: generate images and display
    figure(figsize=(8, 3), dpi=300)
    # Z COMES FROM NORMAL(0, 1)
    mu = 0
    std = 1
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std)) # zero mean and std of one
    z = p.rsample((num_preds,))
    # SAMPLE IMAGES
    with torch.no_grad():
        pred = vae.decoder(z.to(vae.device)).cpu()
    
    img = make_grid(pred).permute(1, 2, 0).numpy() * std + mu + noise
    # PLOT IMAGES
    imshow(img);   

torch.save(vae_model.state_dict(), CHECKPOINTS_DIR / "vae_model.pth")












