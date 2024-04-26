import sys, os, pdb, shutil
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset

from activephasemap.np.neural_process import NeuralProcess
from activephasemap.np.training import NeuralProcessTrainer
from activephasemap.np.utils import context_target_split
from activephasemap.utils.simulators import UVVisExperiment

sys.path.insert(1, '/mmfs1/home/kiranvad/kiranvad/activephasemap-examples/pretrained/')
from helpers import plot_samples, plot_posterior_samples, plot_zgrid_curves

PLOT_DIR = './NPTraining/'
if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.makedirs(PLOT_DIR)
os.makedirs(PLOT_DIR+'itrs/')

ITERATION = 3
EXPT_DATA_DIR = "./data/"
batch_size = 16
num_epochs = 2000
r_dim = 128  # Dimension of representation of context points
z_dim = 2  # Dimension of sampled latent variable
h_dim = 128  # Dimension of hidden layers in encoder and decoder
learning_rate = 5e-3
plot_epochs_freq = 500
print_itr_freq = 10000

class ExperimentalUVVisData(Dataset):
    def __init__(self, iteration, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the data.
        """
        self.expt = UVVisExperiment(iteration, root_dir)
        self.expt.generate(use_spline=True)
        self.xrange = [0,1]

    def __len__(self):
        return len(self.expt.spectra_normalized)

    def __getitem__(self, i):
        x = torch.from_numpy(self.expt.t.astype(np.float32)).unsqueeze(1)
        y = torch.from_numpy(self.expt.spectra_normalized[i].astype(np.float32)).unsqueeze(1)
        return x,y 

# Create dataset
dataset = ExperimentalUVVisData(ITERATION, EXPT_DATA_DIR)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
)
x, y = next(iter(data_loader))
print('Batch data shape for training : ', x.shape, y.shape)
# Visualize data samples
fig, ax = plt.subplots()
for i in np.random.randint(len(dataset), size=100):
    xi, yi = dataset[i]
    ax.plot(xi.cpu().numpy(), yi.cpu().numpy(), c='b', alpha=0.5)
plt.savefig(PLOT_DIR+'data_samples.png')
plt.close()

neuralprocess = NeuralProcess(1, 1, r_dim, z_dim, h_dim).to(device)
# Create a set of 100 target points, with shape 
# (batch_size, num_points, x_dim), which in this case is
# (1, 100, 1)
x_target = torch.linspace(dataset.xrange[0], dataset.xrange[1], 100).reshape(1,100,1).to(device)
with torch.no_grad():
    fig, ax = plt.subplots()
    plot_samples(ax, neuralprocess, x_target, z_dim)
    plt.savefig(PLOT_DIR+'samples_before_training.png')
    plt.close()

optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=learning_rate)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=(3, 47),
                                  num_extra_target_range=(50, 53), 
                                  print_freq=print_itr_freq
                                  )

neuralprocess.training = True
x_plot = torch.linspace(dataset.xrange[0], dataset.xrange[1], steps = 100).reshape(1,100,1).to(device)
np_trainer.train(data_loader, num_epochs, x_plot=x_plot, plot_epoch=plot_epochs_freq, savedir=PLOT_DIR+'/itrs/') 
torch.save(neuralprocess.state_dict(), PLOT_DIR+'uvvis_np.pt')
np.save(PLOT_DIR+'loss.npy', np_trainer.epoch_loss_history) 

neuralprocess.training = False
with torch.no_grad():
    fig, ax = plt.subplots()
    n_smooth = 10
    loss_ = np.convolve(np_trainer.epoch_loss_history, np.ones(n_smooth)/n_smooth, mode='valid')
    ax.plot(np.arange(len(loss_)), loss_)
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

