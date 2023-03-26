import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from skorch.helper import SliceDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
import numpy
import SimpleITK as sitk
from sklearn import datasets
from sklearn.model_selection import train_test_split

import u_net
import utils

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path.cwd() / "TrainingData"
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "segmentation_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]  # images are made smaller to save training time
BATCH_SIZE = 32
N_EPOCHS = 20
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
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

x = dataset.mr_image_list
y = dataset.mask_list
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
loss_function = utils.DiceBCELoss()
unet_model = u_net.UNet(num_classes=1)
optimizer = torch.optim.NAdam(unet_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004) 
 


'''
######### GRID SEARCH   ####################


# Random seed chosen to ensure results are reproducible by using the same initial random weights and biases, 
# and applying dropout rates to the same random embedded categorical features and neurons in the hidden layers
torch.manual_seed(0)

net = NeuralNetClassifier(module=unet_model,
                          max_epochs=20)

iris= datasets.load_iris()
X = iris.data # [105, 1]
y = iris.target # [105 ]

X_train, X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0 )

# define the grid search parameters
param_grid = {
    'optimizer__lr': [0.00001, 0.0001, 0.001, 0.01],
    'optimizer': [optim.SGD, optim.RMSprop, optim.Adam, optim.NAdam],
    'batch_size': [4, 8, 16, 32]
    #'optimizer__momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
}
grid = GridSearchCV(estimator=net, param_grid=param_grid, n_jobs=-1, cv=3)



#d_loader_slice_X = SliceDataset(dataset, idx=0)
#d_loader_slice_y = SliceDataset(dataset, idx=1)
#y_from_ds = numpy.asarray([dataset[i][1] for i in range(len(dataset))])

grid_result = grid.fit(X_train, y_train)

# sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd"))
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
#################################################################

'''

minimum_valid_loss = 10  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

# training loop
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
