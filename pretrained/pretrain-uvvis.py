import sys, os, pdb, shutil
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader
from activephasemap.np.neural_process import NeuralProcess
from activephasemap.np.training import NeuralProcessTrainer
from activephasemap.np.utils import context_target_split

sys.path.append('./helpers.py')
from helpers import UVVisDataset

PLOT_DIR = './UVVis/'
if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.makedirs(PLOT_DIR)
os.makedirs(PLOT_DIR+'/itrs/')
print('Saving the results to %s'%PLOT_DIR)

batch_size = 8
num_context = 40
num_target = 40
num_iterations = 150
x_dim = 1
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 2  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder
learning_rate = 5e-3

# Create dataset
dataset = UVVisDataset(root_dir='./uvvis_data_npy')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
x, y = next(iter(data_loader))
print(x.shape, y.shape)
# Visualize data samples
for i in np.random.randint(len(dataset), size=100):
    xi, yi = dataset[i]
    plt.plot(xi.cpu().numpy(), yi.cpu().numpy(), c='b', alpha=0.5)
plt.savefig(PLOT_DIR+'data_samples.png')
plt.close()

neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
# Create a set of 100 target points, with shape 
# (batch_size, num_points, x_dim), which in this case is
# (1, 100, 1)
x_target = torch.Tensor(np.linspace(0, 1, 100))
x_target = x_target.unsqueeze(1).unsqueeze(0)

with torch.no_grad():
    z_sample = torch.randn((100, z_dim))
    for zi in z_sample:
        mu, _ = neuralprocess.xz_to_y(x_target, zi)
        plt.plot(x_target.numpy()[0], mu.detach().numpy()[0], c='b', alpha=0.5)
    plt.savefig(PLOT_DIR+'samples_before_training.png')

optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=learning_rate)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=(num_context, num_context),
                                  num_extra_target_range=(num_target, num_target), 
                                  print_freq=1000)

neuralprocess.training = True
x_plot = torch.Tensor(np.linspace(0, 1, 100))
x_plot = x_plot.unsqueeze(1).unsqueeze(0)
np_trainer.train(data_loader, num_iterations, 
x_plot=x_plot, plot_after=10, savedir=PLOT_DIR+'/itrs/') 

neuralprocess.training = False

with torch.no_grad():
    z_sample = torch.randn((100, z_dim))
    for zi in z_sample:
        mu, _ = neuralprocess.xz_to_y(x_target, zi)
        plt.plot(x_target.numpy()[0], mu.detach().numpy()[0], 
                c='b', alpha=0.5)

    plt.savefig(PLOT_DIR+'samples_after_training.png')
    plt.close()

    # Extract a batch from data_loader
    fig, axs = plt.subplots(2,5, figsize=(4*5, 4*2))
    for ax in axs.flatten():
        x, y = next(iter(data_loader))
        x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1], 
                                                        num_context, 
                                                        num_target)

        for i in range(200):
            # Neural process returns distribution over y_target
            p_y_pred = neuralprocess(x_context, y_context, x_target)
            # Extract mean of distribution
            mu = p_y_pred.loc.detach()
            ax.plot(x_target.numpy()[0], mu.numpy()[0], alpha=0.05, c='b')

        ax.scatter(x_context[0].numpy(), y_context[0].numpy(), c='tab:red')
        ax.plot(x[0:1].squeeze().numpy(), y[0:1].squeeze().numpy(), c='tab:red')
    plt.savefig(PLOT_DIR+'samples_from_posterior.png')
    plt.close()

    # plot grid
    z1 = torch.linspace(-3,3,10)
    z2 = torch.linspace(-3,3,10)
    fig, axs = plt.subplots(10,10, figsize=(2*10, 2*10))
    for i in range(10):
        for j in range(10):
            z_sample = torch.zeros((1, z_dim))
            z_sample[0,0] = z1[i]
            z_sample[0,1] = z2[j]
            mu, sigma = neuralprocess.xz_to_y(x_target, z_sample)
            mu_, sigma_ = mu.squeeze().numpy(), sigma.squeeze().numpy()
            axs[i,j].plot(x_target.squeeze().numpy(), mu_)
            axs[i,j].fill_between(x_target.squeeze().numpy(), 
            mu_-sigma_, mu_+sigma_,alpha=0.2, color='grey')
            # axs[i,j].set_title('(%.2f, %.2f)'%(z1[i], z2[j]))
    fig.supxlabel('z1')
    fig.supylabel('z2')

    plt.savefig(PLOT_DIR+'samples_in_grid.png')
    plt.close()

# %%
torch.save(neuralprocess.state_dict(), 'uvvis.pt')