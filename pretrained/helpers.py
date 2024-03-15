import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
RNG = np.random.default_rng()
import pdb
import glob

class UVVisDataset(Dataset):
    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the data.
        """
        self.dir = root_dir
        self.files = glob.glob(self.dir+'/*.npz')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            npzfile = np.load(self.files[i])
        except Exception as e:
            print('%s Could not load %s'%(type(e).__name__, self.files[i]))
        wl, I = npzfile['wl'], npzfile['I']
        wl = (wl-min(wl))/(max(wl)-min(wl))
        wl_ = torch.tensor(wl.astype(np.float32)).unsqueeze(1)
        I_ = torch.tensor(I.astype(np.float32)).unsqueeze(1)

        return wl_, I_


def plot_samples(ax, model, x_target, z_dim, num_samples=100):
    z_sample = torch.randn((num_samples, z_dim))
    for zi in z_sample:
        mu, _ = model.xz_to_y(x_target, zi)
        ax.plot(x_target.numpy()[0], mu.detach().numpy()[0], c='b', alpha=0.5)

    return 

def plot_posterior_samples(x_target, data_loader, model):
    fig, axs = plt.subplots(2,5, figsize=(4*5, 4*2))
    for ax in axs.flatten():
        x, y = next(iter(data_loader))
        x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1], 
                                                        num_context, 
                                                        num_target)

        for i in range(200):
            # Neural process returns distribution over y_target
            p_y_pred = model(x_context, y_context, x_target)
            # Extract mean of distribution
            mu = p_y_pred.loc.detach()
            ax.plot(x_target.numpy()[0], mu.numpy()[0], alpha=0.05, c='b')

        ax.scatter(x_context[0].numpy(), y_context[0].numpy(), c='tab:red')
        ax.plot(x[0:1].squeeze().numpy(), y[0:1].squeeze().numpy(), c='tab:red')

    return fig, axs

def plot_zgrid_curves(z_grid, x_target, model):
    z1 = torch.linspace(z_grid[0],z_grid[1],10)
    z2 = torch.linspace(z_grid[0],z_grid[1],10)
    fig, axs = plt.subplots(10,10, figsize=(2*10, 2*10))
    for i in range(10):
        for j in range(10):
            z_sample = torch.zeros((1, 2))
            z_sample[0,0] = z1[i]
            z_sample[0,1] = z2[j]
            mu, sigma = model.xz_to_y(x_target, z_sample)
            mu_, sigma_ = mu.squeeze().numpy(), sigma.squeeze().numpy()
            axs[i,j].plot(x_target.squeeze().numpy(), mu_)
            axs[i,j].fill_between(x_target.squeeze().numpy(), 
            mu_-sigma_, mu_+sigma_, color='tab:blue', alpha=0.5)
            # axs[i,j].set_title('(%.2f, %.2f)'%(z1[i], z2[j]))
    fig.supxlabel('z1')
    fig.supylabel('z2')

    return fig, axs
