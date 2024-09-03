import os, sys, time, shutil, pdb, argparse,json, glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from botorch.utils.sampling import draw_sobol_samples
torch.set_default_dtype(torch.double)

from activephasemap.models.np import NeuralProcess, context_target_split
from activephasemap.models.mlp import MLP
from activephasemap.utils.visuals import get_twod_grid

TRAINING_ITERATIONS = 200 # total iterations for each optimization
NUM_RESTARTS = 8 # number of optimization from random restarts
LEARNING_RATE = 0.1
TARGET_SHAPE = "triangle" # chose from ["sphere", "triangle"]

SAVE_DIR = "./retrosynthesis/%s/"%TARGET_SHAPE
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)

DATA_DIR = './output'
ITERATION = len(glob.glob("./output/comp_model_*.pt"))
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]

design_space_bounds = [(0.0, 87.0), (0.0,11.0)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)
DESIGN_SPACE_DIM = len(design_space_bounds)

# Create a target spectrum
TARGETS_DIR = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pygdm"
if TARGET_SHAPE=="sphere":
    target = np.load(TARGETS_DIR+"/target_sphere.npz")
else:
    target = np.load(TARGETS_DIR+"/triangle_fdtd.npz")
wav = target["x"]
n_domain = len(wav)
t = (wav-min(wav))/(max(wav)-min(wav))
xt = torch.from_numpy(t).to(device).view(1, n_domain, 1)
yt = torch.from_numpy(target["y"]).to(device).view(1, n_domain, 1)

# Load GP and NP models and set them to evaluation mode
print("Using models from iteration %d"%ITERATION)
train_x = torch.load(DATA_DIR+'/train_x_%d.pt'%ITERATION, map_location=device, weights_only=True)
train_z_mean = torch.load(DATA_DIR+'/train_z_mean_%d.pt'%ITERATION, map_location=device, weights_only=True)
train_z_std = torch.load(DATA_DIR+'/train_z_std_%d.pt'%ITERATION, map_location=device, weights_only=True)
mlp = MLP(train_x, train_z_mean, train_z_std)
mlp_state_dict = torch.load(DATA_DIR+'/comp_model_%d.pt'%(ITERATION), map_location=device, weights_only=True)
mlp.load_state_dict(mlp_state_dict)

NP = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
NP.load_state_dict(torch.load(DATA_DIR+'/np_model_%d.pt'%ITERATION, map_location=device, weights_only=True)) 
NP.train(False)

def from_comp_to_spectrum(t, c, comp_model, np_model):
    ci = c.clone().detach().requires_grad_(True)
    z_mu, z_std = comp_model.mlp(ci)
    z_dist = torch.distributions.Normal(z_mu, z_std)
    z = z_dist.sample(torch.Size([128]))
    xt = t.repeat(128, 1, 1).to(device)
    y_samples, _ = np_model.xz_to_y(xt, z.squeeze(1))

    mean_pred = y_samples.mean(dim=0, keepdim=True)
    sigma_pred = y_samples.std(dim=0, keepdim=True)
    mu_ = mean_pred.cpu().squeeze()
    sigma_ = sigma_pred.cpu().squeeze() 

    return mu_, sigma_ 

def min_max_normalize(x):
    x_ = torch.zeros_like(x)
    num_samples, n_domain = x.shape
    for i in range(num_samples):
        x_[i,:] = (x[i,:]-min(x[i,:]))/(max(x[i,:])-min(x[i,:]))
    
    return x_

def simulator(c):
    num_points, dim = c.shape 
    spectra_pred = torch.zeros((num_points, n_domain, 2)).to(device)
    for i in range(num_points):
        ci = c[i,:].reshape(1, DESIGN_SPACE_DIM)
        mu, sigma = from_comp_to_spectrum(xt, ci, mlp, NP)
        spectra_pred[i,:, 0] = mu.squeeze()
        spectra_pred[i,:, 1] = sigma.squeeze()

    target = yt.squeeze().repeat(num_points, 1)
    target_ = min_max_normalize(target)
    mu_ = min_max_normalize(spectra_pred[..., 0])

    loss = torch.nn.functional.mse_loss(mu_, target_, reduction="none").mean(dim=1)

    return loss, spectra_pred

# Initialize using random Sobol sequence sampling
X = draw_sobol_samples(bounds=bounds, n=NUM_RESTARTS, q=1).to(device)
X.requires_grad_(True)

optimizer = torch.optim.Adam([X], lr=LEARNING_RATE)
X_traj, loss_traj, spectra_traj = [], [], []

# run a basic optimization loop
for i in range(TRAINING_ITERATIONS):
    optimizer.zero_grad()
    # this performs batch (num_restrats) evaluation
    losses, spectra = simulator(X.squeeze()) 
    loss = losses.sum()

    loss.backward()  
    optimizer.step()

    # clamp values to the feasible set
    for j, (lb, ub) in enumerate(zip(*bounds)):
        X.data[..., j].clamp_(lb, ub) 

    # store the optimization trajecatory
    X_traj.append(X.detach().clone())
    loss_traj.append(losses.detach().clone())
    spectra_traj.append(spectra.detach().clone())

    if (i + 1) % 10 == 0:
        print(f"Iteration {i+1:>3}/{TRAINING_ITERATIONS:>3} - Loss: {loss.item():>4.3f}")

with torch.no_grad():
    loss_optim, spectra_optim = simulator(X_traj[-1].squeeze())
    print("Optimized composition : ", X_traj[-1].squeeze()[torch.argmin(loss_optim)])
    for i in range(NUM_RESTARTS):
        fig, ax = plt.subplots(figsize=(4, 4))
        mu = spectra_optim[i,:,0].cpu().squeeze().numpy()
        sigma = spectra_optim[i,:,1].cpu().squeeze().numpy()
        ax.plot(target["x"], target["y"], label="Target", color="k", ls='--')
        ax2 = ax.twinx()
        ax2.plot(target["x"], mu, label="Best Estimate", color="k")
        ax2.fill_between(target["x"], mu-sigma, mu+sigma,  
                        color='grey', alpha=0.5, label="Uncertainity"
                        )
        ax.set_title("Loss : %.2f"%loss_optim[i].item())
        plt.savefig(SAVE_DIR+"comparision_%d.png"%i)
        plt.close()

# Compute loss function on a grid for plotting
with torch.no_grad():
    grid_comps = get_twod_grid(30, bounds=bounds.cpu().numpy())
    num_comps,_ = grid_comps.shape
    grid_loss = torch.zeros(num_comps)
    for i in range(num_comps):
        loss, _ = simulator(torch.from_numpy(grid_comps[i,:].reshape(1,-1)).to(device))
        grid_loss[i] = loss.item()

# create result object and save
optim_result = {"X_traj" : torch.stack(X_traj, dim=1).squeeze(),
                "spectra_traj" : torch.stack(spectra_traj, dim=1).squeeze(),
                "loss" : torch.stack(loss_traj, dim=1).squeeze(),
                "spectra" : spectra_optim,
                "target_y" : yt,
                "target_x" : xt,
                "grid_loss" : grid_loss,
                "grid_comps" : grid_comps
                }
torch.save(optim_result, SAVE_DIR+"optim_traj.pkl")