import os, sys, time, shutil, pdb, argparse,json, glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from gpytorch.constraints import Interval
from botorch.utils.sampling import draw_sobol_samples
torch.set_default_dtype(torch.double)
from botorch.utils.transforms import normalize, unnormalize
from botorch.optim.initializers import initialize_q_batch_nonneg
from gpytorch.distributions import MultivariateNormal as mvn

from activephasemap.models.np import NeuralProcess, context_target_split
from activephasemap.models.gp import MultiTaskGP
from activephasemap.utils.simulators import GNPPhases
from activephasemap.utils.settings import initialize_model 
from activephasemap.utils.settings import get_twod_grid


TRAINING_ITERATIONS = 100 # total iterations for each optimization
NUM_RESTARTS = 8 # number of optimization from random restarts
LEARNING_RATE = 0.1
TARGET_SHAPE = "triangle" # chose from ["sphere", "triangle"]

SAVE_DIR = "./retrosynthesis/%s/"%TARGET_SHAPE
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)

gp_model_args = {"model":"gp", "num_epochs" : 1, "learning_rate" : 1e-3, "verbose": 1}
DATA_DIR = './output'
ITERATION = len(glob.glob("./data/gp_model_*.npy"))
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]

design_space_bounds = [(0.0, 75.0),
                       (0.0, 75.0),
                       (0.0, 75.0),
                       (0.0, 75.0), 
                       (0.0, 11.0),
                       ]
DESIGN_SPACE_DIM = len(design_space_bounds)
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

# Create a target spectrum
target = np.load("../pygdm/target_%s.npz"%TARGET_SHAPE)
wav = target["x"]
n_domain = len(wav)
t = (wav-min(wav))/(max(wav)-min(wav))
xt = torch.from_numpy(t).to(device).view(1, n_domain, 1)
yt = torch.from_numpy(target["y"]).to(device).view(1, n_domain, 1)

# Load GP and NP models and set them to evaluation mode
train_x = torch.load(DATA_DIR+'/train_x_%d.pt'%ITERATION, map_location=device)
train_y = torch.load(DATA_DIR+'/train_y_%d.pt'%ITERATION, map_location=device)
train_y_std = torch.load(DATA_DIR+'/train_y_std%d.pt'%ITERATION, map_location=device)
normalized_x = normalize(train_x, bounds).to(train_x)
print(normalized_x.max(), normalized_x.min())
GP = MultiTaskGP(normalized_x, train_y, gp_model_args, DESIGN_SPACE_DIM, N_LATENT, train_y_std)
gp_state_dict = torch.load(DATA_DIR+'/gp_model_%d.pt'%ITERATION, map_location=device)
GP.load_state_dict(gp_state_dict)
GP.train(False)

NP = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
NP.load_state_dict(torch.load(DATA_DIR+'/np_model_%d.pt'%ITERATION, map_location=device)) 
NP.train(False)

def from_comp_to_spectrum(gp_model, np_model, t, c, bounds):
    normalized_x = normalize(c, bounds)
    posterior = gp_model.posterior(normalized_x)
    mu = []
    for _ in range(128):
        z = posterior.rsample().squeeze(0)
        mu_i, _ = np_model.xz_to_y(t, z)
        mu.append(mu_i)

    mean_pred = torch.cat(mu).mean(dim=0, keepdim=True)
    sigma_pred = torch.cat(mu).std(dim=0, keepdim=True)

    return mean_pred, sigma_pred 

def simulator(c):
    num_points, dim = c.shape 
    spectra_pred = torch.zeros((num_points, n_domain, 2)).to(device)
    for i in range(num_points):
        ci = c[i,:].reshape(1, DESIGN_SPACE_DIM)
        mu, sigma = from_comp_to_spectrum(GP, NP, xt, ci, bounds)
        spectra_pred[i,:, 0] = mu.squeeze()
        spectra_pred[i,:, 1] = sigma.squeeze()

    target = yt.squeeze().repeat(num_points, 1)

    loss = torch.nn.functional.mse_loss(spectra_pred[..., 0], target, reduction="none").mean(dim=1)

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
optim_result = {"X_traj" : torch.stack(X_traj, dim=1).squeeze(),
                "spectra_traj" : torch.stack(spectra_traj, dim=1).squeeze(),
                "loss" : torch.stack(loss_traj, dim=1).squeeze(),
                "spectra" : spectra_optim,
                "target_y" : yt,
                "target_x" : xt,
                }
torch.save(optim_result, SAVE_DIR+"optim_traj.pkl")