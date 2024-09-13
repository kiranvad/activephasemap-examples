import os, sys, time, shutil, pdb, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)

from activephasemap.models.mlp import MLP
from activephasemap.models.np import NeuralProcess
from activephasemap.utils.visuals import get_twod_grid
from activephasemap.utils.simulators import UVVisExperiment

DATA_DIR = "../output"
ITERATION = 7
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]

# Load MLP model for q(z|c)
train_x = torch.load(DATA_DIR+'/train_x_%d.pt'%ITERATION, map_location=device)
train_z_mean = torch.load(DATA_DIR+'/train_z_mean_%d.pt'%ITERATION, map_location=device)
train_z_std = torch.load(DATA_DIR+'/train_z_std_%d.pt'%ITERATION, map_location=device)
comp_model = MLP(train_x, train_z_mean, train_z_std)
mlp_state_dict = torch.load(DATA_DIR+'/comp_model_%d.pt'%(ITERATION), map_location=device)
comp_model.load_state_dict(mlp_state_dict)

# Load trained NP model for p(y|z)
np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
np_model.load_state_dict(torch.load(DATA_DIR+'/np_model_%d.pt'%(ITERATION), map_location=device))

# Load experimental data
design_space_bounds = [(0.0, 87.0), (0.0,11.0)]
expt = UVVisExperiment(design_space_bounds, "../data/")
expt.read_iter_data(ITERATION)
expt.generate(use_spline=True)

# Create a grid of composition space
N_GRID_SPACING = 4
grid_comps = get_twod_grid(N_GRID_SPACING, expt.bounds.cpu().numpy())

def from_comp_to_spectrum(t, c, comp_model, np_model):
    """ Predict a spectrum given composition using the composite model


    """
    ci = torch.tensor(c).to(device)
    z_mu, z_std = comp_model.mlp(ci)
    print(comp_model)
    z_mu = comp_model.mu_scaler.inverse_transform(z_mu)
    z_std = comp_model.std_scaler.inverse_transform(z_std)
    z_dist = torch.distributions.Normal(z_mu, z_std)
    z = z_dist.sample(torch.Size([100]))
    t = torch.from_numpy(t).repeat(100, 1, 1).to(device)
    t = torch.swapaxes(t, 1, 2)
    y_samples, _ = np_model.xz_to_y(t, z)

    mean_pred = y_samples.mean(dim=0, keepdim=True)
    sigma_pred = y_samples.std(dim=0, keepdim=True)
    mu_ = mean_pred.cpu().squeeze()
    sigma_ = sigma_pred.cpu().squeeze() 
    
    return mu_, sigma_, z_mu, z_std 


@torch.no_grad()
def plot_mlp_grid(grid_comps):
    """ Plot predictions on a grid

    This provides a qualitative understanding of current model 
    on training and test data.
    Useful to understand leanred model of uncertainity.
    """

    print("Creating plots to visualize training data predictions...")
    iter_plot_dir = "preds_%d/"%ITERATION
    if os.path.exists(iter_plot_dir):
        shutil.rmtree(iter_plot_dir)
    os.makedirs(iter_plot_dir)

    num_samples, c_dim = grid_comps.shape
    for i in range(num_samples):
        print("Plotting %d..."%i,end='\r', flush=True)
        fig, axs = plt.subplots(1,3, figsize=(3*4, 4))

        # Plot MLP model predictions of the spectra
        mu, sigma, z_mu, z_std = from_comp_to_spectrum(expt.t,
                                                       grid_comps[i,:], 
                                                       comp_model, 
                                                       np_model
                                                       )
        axs[0].plot(expt.wl, mu)
        minus = (mu-sigma)
        plus = (mu+sigma)
        axs[0].fill_between(expt.wl, minus, plus, color='grey')
        axs[0].scatter(expt.wl, expt.spectra_normalized[i,:], color='k', s=10)
        axs[0].set_title("(MLP) time : %d conc : %.2f"%(expt.comps[i,1], expt.comps[i,0]))
        
        # Plot the Z values trained MLP predictions
        mlp_pred = torch.distributions.Normal(z_mu, z_std).sample(torch.Size([100]))
        axs[1].violinplot(mlp_pred.cpu().numpy(), showmeans=True)

        # Plot the location of the current prediction wrto train data
        axs[2].scatter(train_x[:,0], train_x[:,1], color="grey", facecolor="none")
        axs[2].scatter(grid_comps[i,0], grid_comps[i,1], color="tab:red", s=50)

        plt.savefig(iter_plot_dir+'%d.png'%(i))
        plt.close()

plot_mlp_grid(grid_comps)