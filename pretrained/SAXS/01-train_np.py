import sys, os, pdb, shutil, json, pdb
from math import pi
import numpy as np
import matplotlib.pyplot as plt

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
import torch.utils.data as tdata

from activephasemap.models.np import NeuralProcess, train_neural_process
from activephasemap.pretrained.helpers import *

PLOT_DIR = './01b/'
if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.makedirs(PLOT_DIR)

batch_size = 2
r_dim = 128  # Dimension of representation of context points
z_dim = 8  # Dimension of sampled latent variable
h_dim = 128  # Dimension of hidden layers in encoder and decoder
n_blocks = 5 # Number of neural network layers in each encoder and decoder
learning_rate = 1e-3
pos_basis = "fourier" # Basis functions to use for embedding

num_epochs = 100
plot_epochs_freq = 10

# Create dataset
dataset = SAXSLogLog(root_dir='/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/SAXS/')
train_dataset, eval_dataset = tdata.random_split(dataset, [0.8, 0.2])
collate_fn=lambda x: tuple(x_.to(device) for x_ in tdata.dataloader.default_collate(x))
train_data_loader = tdata.DataLoader(train_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True,
                                     collate_fn = collate_fn
                                )
eval_data_loader = tdata.DataLoader(eval_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True,
                                     collate_fn = collate_fn
                                )                                

x, y = next(iter(train_data_loader))
print('Batch data shape for training : ', x.shape, y.shape)

# Visualize data samples
fig, ax = plot_dataset_samples(dataset)
plt.savefig(PLOT_DIR+'data_samples.png')
plt.close()

neuralprocess = NeuralProcess(r_dim, z_dim, h_dim, n_blocks, pos_basis).to(device)
# Create a set of 100 target points, with shape 
# (batch_size, num_points, x_dim), which in this case is (1, 100, 1)
x_target = torch.linspace(dataset.xrange[0], 
                          dataset.xrange[1], 
                          dataset.n_domain
                          ).reshape(1, dataset.n_domain, 1).to(device)

with torch.no_grad():
    if isinstance(dataset, SAXSLogLog):
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(1,2, figsize=(4*2, 4))
    plot_samples(ax, dataset, neuralprocess, x_target, z_dim)
    plt.savefig(PLOT_DIR+'samples_before_training.png')
    plt.close()

# Compute evaluation MSE of the model
@torch.no_grad()
def evaluate_np_model(model, data_loader):
    mse = 0
    for i, data in enumerate(data_loader):
        x, y = data
        n_domain = x.shape[1]
        inds = torch.randint(0, n_domain, (15,))
        p_y_pred = model(x[:,inds,:], y[:,inds,:], x)
        mu_pred = p_y_pred.mean 
        mse += ((y-mu_pred)**2).sum()

    return mse/len(data_loader)

# Train neural process model
neuralprocess.training = True
optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=learning_rate)
epoch_loss_train, epoch_loss_eval = [], []
for epoch in range(num_epochs+1):
    neuralprocess, optimizer, loss_value = train_neural_process(neuralprocess, train_data_loader, optimizer)
    eval_loss = evaluate_np_model(neuralprocess, eval_data_loader)
    print("Epoch: %d, Train loss value : %2.4f, Eval Loss value : %2.4f"%(epoch, loss_value, eval_loss))
    epoch_loss_train.append(loss_value)
    epoch_loss_eval.append(eval_loss)

    if (epoch)%plot_epochs_freq==0:
        torch.save(neuralprocess.state_dict(), PLOT_DIR+"model.pt")
        with torch.no_grad():
            if isinstance(dataset, SAXSLogLog):
                fig, ax = plt.subplots()
            else:
                fig, ax = plt.subplots(1,2, figsize=(4*2, 4))
            plot_samples(ax, dataset, neuralprocess, x_target, z_dim)
            plt.savefig(PLOT_DIR+'itr_%d.png'%(epoch))
            plt.close()

torch.save(neuralprocess.state_dict(), PLOT_DIR+'model.pt')
np.save(PLOT_DIR+'train_loss.npy', epoch_loss_train) 
np.save(PLOT_DIR+'eval_loss.npy', epoch_loss_eval)

neuralprocess.training = False
with torch.no_grad():
    fig, ax = plt.subplots()
    color="tab:blue"
    ax.plot(np.arange(len(epoch_loss_train)), epoch_loss_train, color=color)
    ax.set_ylabel('Train loss', color=color)
    ax.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()
    color = 'tab:red'
    ax2.plot(np.arange(len(epoch_loss_eval)), epoch_loss_eval, color=color)
    ax2.set_ylabel('Eval MSE', color=color) 
    ax2.tick_params(axis='y', labelcolor=color)

    ax.set_xlabel('Epochs')
    plt.savefig(PLOT_DIR+'loss.png')
    plt.close()

    # Plot samples from the trained model
    if isinstance(dataset, SAXSLogLog):
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(1,2, figsize=(4*2, 4))
    plot_samples(ax, dataset, neuralprocess, x_target, z_dim)
    plt.savefig(PLOT_DIR+'samples_after_training.png')
    plt.title("Mean training loss : %2.4f"%loss_value)
    plt.close()

    # Plot curve fitting-like samples from posteriors
    plot_posterior_samples(x_target, dataset, neuralprocess)
    plt.savefig(PLOT_DIR+'samples_from_posterior.png')
    plt.title("Mean MSE on evaluation : %2.4f"%eval_loss)
    plt.close()


