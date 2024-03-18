import sys, os, pdb, shutil
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader
from activephasemap.np.neural_process import NeuralProcess
from activephasemap.np.training import NeuralProcessTrainer
from activephasemap.np.datasets import SineData

sys.path.append('./helpers.py')
from helpers import *

PLOT_DIR = './SINES/'
if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.makedirs(PLOT_DIR)
os.makedirs(PLOT_DIR+'/itrs/')
print('Saving the results to %s'%PLOT_DIR)

batch_size = 8
num_epochs = 10
r_dim = 50  # Dimension of representation of context points
z_dim = 2  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder
learning_rate = 5e-3
plot_epochs_freq = 2
print_itr_freq = 1000

# Create dataset
dataset = SineData(amplitude_range=(-1., 1.),
                   shift_range=(-.5, .5),
                   num_samples=2000)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
x, y = next(iter(data_loader))
print('Batch data shape for training : ', x.shape, y.shape)
# Visualize data samples
fig, ax = plt.subplots()
for i in np.random.randint(len(dataset), size=100):
    xi, yi = dataset[i]
    ax.plot(xi.cpu().numpy(), yi.cpu().numpy(), c='b', alpha=0.5)
plt.savefig(PLOT_DIR+'data_samples.png')
plt.close()

neuralprocess = NeuralProcess(1, 1, r_dim, z_dim, h_dim)
# Create a set of 100 target points, with shape 
# (batch_size, num_points, x_dim), which in this case is
# (1, 100, 1)
x_target = torch.Tensor(np.linspace(-3, 3, 100))
x_target = x_target.unsqueeze(1).unsqueeze(0)

with torch.no_grad():
    fig, ax = plt.subplots()
    plot_samples(ax, neuralprocess, x_target, z_dim)
    plt.savefig(PLOT_DIR+'samples_before_training.png')
    plt.close()

optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=learning_rate)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=(3, 50),
                                  num_extra_target_range=(1, 50), 
                                  print_freq=print_itr_freq)

neuralprocess.training = True
x_plot = torch.Tensor(np.linspace(-3, 3, 100))
x_plot = x_plot.unsqueeze(1).unsqueeze(0)
np_trainer.train(data_loader, num_epochs, x_plot=x_plot, plot_epoch=plot_epochs_freq, savedir=PLOT_DIR+'/itrs/') 

neuralprocess.training = False

with torch.no_grad():
    fig, ax = plt.subplots()
    ax.plot(np.arange(num_epochs), np_trainer.epoch_loss_history)
    plt.savefig(PLOT_DIR+'loss.png')
    plt.close()

    # Plot samples from the trained model
    fig, ax = plt.subplots()
    plot_samples(ax, neuralprocess, x_target, z_dim)
    plt.savefig(PLOT_DIR+'samples_after_training.png')
    plt.close()

    # Plot curve fitting-like samples from posteriors
    plot_posterior_samples(x_target, data_loader, neuralprocess)
    plt.savefig(PLOT_DIR+'samples_from_posterior.png')
    plt.close()

    # plot grid of possible z-values
    if z_dim==2:
        plot_zgrid_curves([-5,5], x_target, neuralprocess)
        plt.savefig(PLOT_DIR+'samples_in_grid.png')
        plt.close()

torch.save(neuralprocess.state_dict(), 'uvvis.pt')