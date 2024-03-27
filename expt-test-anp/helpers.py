import os, sys, time, shutil, pdb
import numpy as np 

import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps 
from matplotlib.cm import ScalarMappable

import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from botorch.utils.transforms import normalize, unnormalize
from activephasemap.utils.settings import get_twod_grid
from activephasemap.utils.visuals import _inset_spectra, MinMaxScaler, scaled_tickformat


def get_contex_target(x,y, max_num_context):
    x_ = x.to(device)
    y_ = y.to(device)
    num_context = int(np.random.rand()*(max_num_context - 3) + 3)
    num_target = int(np.random.rand()*(max_num_context - num_context))
    num_total_points = x_.shape[1]
    idx = torch.randperm(num_total_points)
    target_x = x_[:, idx[:num_target + num_context],:]
    target_y = y_[:, idx[:num_target + num_context],:]
    context_x = x_[:, idx[:num_context]]
    context_y = y_[:, idx[:num_context]]

    return context_x, context_y, target_x, target_y



def from_comp_to_spectrum(test_function, gp_model, np_model, c, rep_dim):
    with torch.no_grad():
        t_ = test_function.sim.t
        t = torch.from_numpy(t_).to(device)
        t = t.repeat(c.shape[0]).view(c.shape[0], len(t_), 1)

        c = torch.tensor(c).to(device)
        gp_model.eval()
        normalized_x = normalize(c, test_function.bounds.to(c))
        posterior = gp_model.posterior(normalized_x)

        rzc = posterior.rsample()
        rc, zc = rzc[:, :, :rep_dim], rzc[:, :, rep_dim:]
        pdb.set_trace()
        dist = np_model.decoder(rc, zc, t)

        return dist.loc, dist.scale

def plot_gpmodel_grid(ax, test_function, itr, **kwargs):
    bounds = test_function.bounds.cpu().numpy()
    num_grid_spacing = kwargs.pop("num_grid_spacing", 10)
    c1 = np.linspace(bounds[0,0], bounds[1,0], num=num_grid_spacing)
    c2 = np.linspace(bounds[0,1], bounds[1,1], num=num_grid_spacing)
    scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
    scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
    if kwargs.pop("scale_axis", True):
        ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
        ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    with torch.no_grad():
        for i in range(num_grid_spacing):
            for j in range(num_grid_spacing):
                ci = np.array([c1[i], c2[j]]).reshape(1, 2)
                rep_dim = kwargs.pop("np_rep_dim", 6)
                mu, sigma = from_comp_to_spectrum(test_function, itr.gp_model, itr.np_model, ci, rep_dim)
                mu_ = mu.cpu().squeeze().numpy()
                sigma_ = sigma.cpu().squeeze().numpy()
                norm_ci = np.array([scaler_x.transform(c1[i]), scaler_y.transform(c2[j])])
                _inset_spectra(norm_ci, test_function.sim.t, mu_, sigma_, ax, **kwargs)

    ax.set_title("Predicted Spectra")

    return 
    
def plot_iteration(itr, test_function, hyper_params):
    layout = [['A1','A2', 'C', 'C'], 
              ['B1', 'B2', 'C', 'C']
              ]
    
    # plot selected points
    C_train = test_function.sim.points
    C_grid = get_twod_grid(20, test_function.bounds.cpu().numpy())
    fig, axs = plt.subplot_mosaic(layout, figsize=(4*4, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if torch.is_tensor(itr.train_x):
        x_ = itr.train_x.cpu().numpy()
    else:
        x_ = itr.train_x
    axs['A1'].scatter(x_[:,0], x_[:,1], marker='x', color='k')  
    axs['A1'].set_title('Sampled Comp.')
    axs['A1'].set_xlim([test_function.bounds[0,0], test_function.bounds[1,0]])
    axs['A1'].set_ylim([test_function.bounds[0,1], test_function.bounds[1,1]])

    # plot acqf
    normalized_C_grid = normalize(torch.tensor(C_grid).to(device), test_function.bounds.to(device))
    with torch.no_grad():
        acq_values = itr.acquisition(normalized_C_grid.reshape(len(C_grid),1,2)).cpu().numpy()
    cmap = colormaps["magma"]
    norm = Normalize(vmin=min(acq_values), vmax = max(acq_values))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    axs['A2'].tricontourf(C_grid[:,0], C_grid[:,1], acq_values, cmap=cmap, norm=norm)
    divider = make_axes_locatable(axs["A2"])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel('Acqusition value')
    axs['A2'].set_xlabel('C1', fontsize=20)
    axs['A2'].set_ylabel('C2', fontsize=20) 

    gp_loss_ = np.convolve(itr.gp_loss, np.ones(100)/100, mode='valid')
    fig, ax = plt.subplots()
    axs['B2'].plot(np.arange(len(gp_loss_)), gp_loss_)
    axs['B2'].set_xlabel('Epochs')
    axs['B2'].set_ylabel('Loss') 
    axs['B2'].set_title('GP')

    np_loss_ = np.convolve(itr.np_loss, np.ones(100)/100, mode='valid')
    fig, ax = plt.subplots()
    axs['B1'].plot(np.arange(len(np_loss_)), np_loss_)
    axs['B1'].set_xlabel('Epochs')
    axs['B1'].set_ylabel('NP-Loss') 
    axs['B1'].set_title('NP')

    plot_gpmodel_grid(axs['C'], test_function, itr, np_rep_dim=hyper_params["rep_dim"], show_sigma=False)

    return fig, axs

def plot_model_accuracy(direc, itr, test_function):
    """ Plot accuract of model predictions of experimental data

    """
    num_samples, c_dim = test_function.sim.comps.shape
    if os.path.exists(direc+'preds/'):
        shutil.rmtree(direc+'preds/')
    os.makedirs(direc+'preds/')
    for i in range(num_samples):
        fig, ax = plt.subplots()
        ci = test_function.sim.comps[i,:].reshape(1, c_dim)
        mu, sigma = from_comp_to_spectrum(test_function, gp_model, np_model, c)
        mu_ = mu.cpu().squeeze()
        sigma_ = sigma.cpu().squeeze()
        ax.plot(test_function.sim.t, mu_, label="GP pred.")
        ax.fill_between(test_function.sim.t,mu_-sigma_,mu_+sigma_, color='grey', label="GP Unc.")
        ax.scatter(test_function.sim.t, test_function.sim.F[i], color='k')
        plt.savefig(direc+'preds/%d.png'%(i))
        plt.close()