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
TARGET_SHAPE_ID = 0 # chose from [0 - "sphere", 1 - "triangle"]

TARGET_SHAPES = ["sphere", "triangle"]
SAVE_DIR = "./retrosynthesis/%s/"%TARGET_SHAPES[TARGET_SHAPE_ID]
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
if TARGET_SHAPE_ID==0:
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
mlp.train(False)

NP = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
NP.load_state_dict(torch.load(DATA_DIR+'/np_model_%d.pt'%ITERATION, map_location=device, weights_only=True)) 
NP.train(False)

def min_max_normalize(x):
    min_x = x.min(dim=1).values 
    max_x = x.max(dim=1).values
    x_norm = (x - min_x[:,None])/((max_x-min_x)[:,None])
    
    return x_norm

class Simulator(torch.nn.Module):
    def __init__(self, xt, c2z, z2y):
        super().__init__()
        self.c_to_z = c2z 
        self.z_to_y = z2y
        self.t = xt

    def from_comp_to_spectrum(self, comp):
        z_mu, z_std = self.c_to_z.mlp(comp)
        z_dist = torch.distributions.Normal(z_mu, z_std)
        z = z_dist.rsample(torch.Size([128]))
        y_samples, _ = self.z_to_y.xz_to_y(self.t.repeat(128, 1, 1).to(device), z.squeeze(1))

        mean_pred = y_samples.mean(dim=0, keepdim=True).squeeze()
        sigma_pred = y_samples.std(dim=0, keepdim=True).squeeze()

        return mean_pred, sigma_pred  

    def forward(self, batch_comp):
        num_points, dim = batch_comp.shape 
        spectra_pred = torch.zeros((num_points, n_domain, 2), requires_grad=True).to(device)
        for i in range(num_points):
            ci = batch_comp[i,:].view(1, dim)
            mu, sigma = self.from_comp_to_spectrum(ci)
            spectra_pred[i,:, 0] = mu
            spectra_pred[i,:, 1] = sigma 

        return spectra_pred                   

def mse_loss(y_pred):
    num_points, _ = y_pred.shape
    target = yt.squeeze().repeat(num_points, 1)
    target_ = min_max_normalize(target)
    mu_ = min_max_normalize(y_pred)

    loss = torch.nn.functional.mse_loss(mu_, target_, reduction="none").mean(dim=1)

    return loss    

sim = Simulator(xt, mlp, NP).to(device)

# Initialize using random Sobol sequence sampling
X = draw_sobol_samples(bounds=bounds, n=NUM_RESTARTS, q=1).to(device)
X.requires_grad_(True)

optimizer = torch.optim.Adam([X], lr=LEARNING_RATE)
X_traj, loss_traj, spectra_traj = [], [], []

# run a basic optimization loop
for i in range(TRAINING_ITERATIONS):
    optimizer.zero_grad()
    # this performs batch (num_restrats) evaluation
    spectra = sim(X.squeeze())
    losses = mse_loss(spectra[...,0]) 
    loss = losses.sum()

    loss.backward()  
    optimizer.step()
    
    # clamp values to the feasible set
    for j, (lb, ub) in enumerate(zip(*bounds)):
        X.data[..., j].clamp_(lb, ub) 

    # store the optimization trajecatory
    # clone and detaching is importat to not meddle with the autograd
    X_traj.append(X.clone().detach())
    loss_traj.append(losses.clone().detach())
    spectra_traj.append(spectra.clone().detach())

    if (i + 1) % 10 == 0:
        print(f"Iteration {i+1:>3}/{TRAINING_ITERATIONS:>3} - Loss: {loss.item():>4.3f}")

with torch.no_grad():
    spectra_optim = sim(X_traj[-1].squeeze())
    loss_optim = mse_loss(spectra_optim[...,0])
    print("Optimized composition : ", X_traj[-1].squeeze()[torch.argmin(loss_optim)])
    X_traj = torch.stack(X_traj, dim=1).squeeze()
    for i in range(NUM_RESTARTS):
        fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
        mu = spectra_optim[i,:,0].cpu().squeeze().numpy()
        sigma = spectra_optim[i,:,1].cpu().squeeze().numpy()
        axs[0].plot(target["x"], target["y"], label="Target", color="k", ls='--')
        ax2 = axs[0].twinx()
        ax2.plot(target["x"], mu, label="Best Estimate", color="k")
        ax2.fill_between(target["x"], mu-sigma, mu+sigma,  
                        color='grey', alpha=0.5, label="Uncertainity"
                        )
        axs[0].set_title("Loss : %.2f"%loss_optim[i].item())
        traj = X_traj.cpu().numpy()[i,:,:]
        axs[1].plot(traj[:,0], traj[:,1],
                    lw=2,c='k', label="Trajectory"
                    )
        axs[1].scatter(traj[0,0], traj[0,1],
                       s=100,c='k',marker='.',
                       zorder=10,lw=2, label="Initial"
                       )
        axs[1].scatter(traj[-1,0], traj[-1,1],
                       s=100,c='k',marker='+',
                       zorder=10,lw=2, label="Final"
                       )
        axs[1].set_xlim(*design_space_bounds[0])
        axs[1].set_ylim(*design_space_bounds[1])
        axs[1].legend()
        plt.savefig(SAVE_DIR+"comparision_%d.png"%i)
        plt.close()

# Compute loss function on a grid for plotting
with torch.no_grad():
    grid_comps = get_twod_grid(30, bounds=bounds.cpu().numpy())
    grid_spectra = sim(torch.from_numpy(grid_comps).to(device))
    grid_loss = mse_loss(grid_spectra[...,0])
    print(grid_loss.shape)

# create result object and save
optim_result = {"X_traj" : X_traj,
                "spectra_traj" : torch.stack(spectra_traj, dim=1).squeeze(),
                "loss" : torch.stack(loss_traj, dim=1).squeeze(),
                "spectra" : spectra_optim,
                "target_y" : yt,
                "target_x" : xt,
                "grid_loss" : grid_loss,
                "grid_comps" : grid_comps
                }
torch.save(optim_result, SAVE_DIR+"optim_traj.pkl")