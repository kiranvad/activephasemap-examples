import sys, os, pdb, shutil, json, pdb
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from activephasemap.models.np import NeuralProcess, train_neural_process
from activephasemap.pretrained.helpers import *
import tempfile

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

num_epochs = 500
plot_epochs_freq = 50
print_itr_freq = 1000

# Create dataset
# dataset = SAXSPorod(root_dir='/mmfs1/home/kiranvad/cheme-kiranvad/sas-55m-20k',  n_sub_sample=250)
dataset = SAXSLogLog(root_dir='/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/SAXS/')

data_loader = DataLoader(dataset, 
                         batch_size=batch_size, shuffle=True,
                         collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
                        )

x, y = next(iter(data_loader))
print('Batch data shape for training : ', x.shape, y.shape)

# Visualize data samples
fig, ax = plot_dataset_samples(dataset)
plt.savefig(PLOT_DIR+'data_samples.png')
plt.close()

neuralprocess = NeuralProcess(r_dim, z_dim, h_dim, n_blocks).to(device)
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

# Train neural orocess model
neuralprocess.training = True
optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=learning_rate)
epoch_loss = []
for epoch in range(num_epochs+1):
    neural_process, optimizer, loss_value = train_neural_process(neuralprocess, data_loader,optimizer)

    if (epoch)%plot_epochs_freq==0:
        torch.save(neural_process.state_dict(), PLOT_DIR+"model.pt")
        with torch.no_grad():
            if isinstance(dataset, SAXSLogLog):
                fig, ax = plt.subplots()
            else:
                fig, ax = plt.subplots(1,2, figsize=(4*2, 4))
            plot_samples(ax, dataset, neural_process, x_target, z_dim)
            plt.savefig(PLOT_DIR+'itr_%d.png'%(epoch))
            plt.close()

    print("Epoch: %d, Loss value : %2.4f"%(epoch, loss_value))
    epoch_loss.append(loss_value)

torch.save(neuralprocess.state_dict(), PLOT_DIR+'model.pt')
np.save(PLOT_DIR+'loss.npy', epoch_loss) 

neuralprocess.training = False
with torch.no_grad():
    fig, ax = plt.subplots()
    n_smooth = 10
    loss_ = np.convolve(epoch_loss, np.ones(n_smooth)/n_smooth, mode='valid')
    ax.plot(np.arange(len(loss_)), loss_)
    plt.savefig(PLOT_DIR+'loss.png')
    plt.close()

    # Plot samples from the trained model
    if isinstance(dataset, SAXSLogLog):
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(1,2, figsize=(4*2, 4))
    plot_samples(ax, dataset, neuralprocess, x_target, z_dim)
    plt.savefig(PLOT_DIR+'samples_after_training.png')
    plt.close()

    # Plot curve fitting-like samples from posteriors
    plot_posterior_samples(x_target, dataset, neuralprocess)
    plt.savefig(PLOT_DIR+'samples_from_posterior.png')
    plt.close()


