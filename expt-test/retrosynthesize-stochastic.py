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
from torch.distributions import Normal

from activephasemap.models.np import NeuralProcess, context_target_split
from activephasemap.utils.simulators import GNPPhases
from activephasemap.utils.settings import initialize_model 
from activephasemap.utils.settings import get_twod_grid


TRAINING_ITERATIONS = 200 # total iterations for each optimization
NUM_RESTARTS = 8 # number of optimization from random restarts
LEARNING_RATE = 0.1

SAVE_DIR = "./retrosynthesis/"
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)

gp_model_args = {"model":"gp", "num_epochs" : 1, "learning_rate" : 1e-3, "verbose": 1}
DATA_DIR = './output'
ITERATION = len(glob.glob("./data/spectra_*.npy"))-1
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]
DESIGN_SPACE_DIM = 2
design_space_bounds = [(0.0, 7.38), (0.0,7.27)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

# Create a target spectrum
sim = GNPPhases("../AuNP/gold_nano_grid/")
target_comp = np.array([4.4, 5.8])
target = sim.simulate(target_comp) 
n_domain = target.shape[0]
xt = torch.from_numpy(sim.t).to(device).view(1, n_domain, 1)
yt = torch.from_numpy(target).to(device).view(1, n_domain, 1)

# Load GP and NP models and set them to evaluation mode
train_x = torch.load(DATA_DIR+'/train_x_%d.pt'%ITERATION, map_location=device)
train_y = torch.load(DATA_DIR+'/train_y_%d.pt'%ITERATION, map_location=device)
normalized_x = normalize(train_x, bounds).to(train_x)
print(normalized_x.max(), normalized_x.min())
GP = initialize_model(normalized_x, train_y, gp_model_args, DESIGN_SPACE_DIM, N_LATENT, device)
gp_state_dict = torch.load(DATA_DIR+'/gp_model_%d.pt'%ITERATION, map_location=device)
GP.load_state_dict(gp_state_dict)
GP.train(False)

NP = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
NP.load_state_dict(torch.load(DATA_DIR+'/np_model_%d.pt'%ITERATION, map_location=device)) 
NP.train(False)

def simulator(c):
    num_points, dim = c.shape 
    num_z_samples = 16
    normalized_x = normalize(c, bounds)
    posterior = GP.posterior(normalized_x)
    z_samples = posterior.rsample(torch.Size([num_z_samples]))
    spectra_pred = torch.zeros((num_points, n_domain, 2))
    for i in range(num_points):
        zi = z_samples[:,i,:].squeeze(0)
        yi_samples = []
        for j in range(num_z_samples):
            zij = zi[j,:]
            yi_samples.append(NP.xz_to_y(xt, zij)[0])
        spectra_pred[i,:, 0] = torch.cat(yi_samples).mean(dim=0).squeeze()
        spectra_pred[i,:, 1] = torch.cat(yi_samples).std(dim=0).squeeze()

    target = yt.squeeze().repeat(num_points, 1)
    loss = torch.nn.functional.mse_loss(spectra_pred[..., 0], target, reduction="none")

    return loss.mean(dim=1), spectra_pred

# # Initialize points using botorch approach
# # generate a large number of random samples
# Xraw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(100 * NUM_RESTARTS,1, DESIGN_SPACE_DIM)
# Yraw,_ = simulator(Xraw.squeeze())  # evaluate the acquisition function on these q-batches
# print(Xraw.shape, Yraw.shape)

# # apply the heuristic for sampling promising initial conditions
# X = initialize_q_batch_nonneg(Xraw, Yraw, NUM_RESTARTS)

# Initialize using random Sobol sequence sampling
X = draw_sobol_samples(bounds=bounds, n=NUM_RESTARTS, q=1).to(device)

# we'll want gradients for the input
X.requires_grad_(True)

# set up the optimizer, make sure to only pass in the candidate set here
optimizer = torch.optim.Adam([X], lr=LEARNING_RATE)
X_traj, loss_traj = [], []  # we'll store the results

# run a basic optimization loop
for i in range(TRAINING_ITERATIONS):
    optimizer.zero_grad()
    # this performs batch evaluation, so this is an N-dim tensor
    losses,_ = simulator(X.squeeze()) 
    loss = losses.sum()

    loss.backward()  # perform backward pass
    optimizer.step()  # take a step

    # clamp values to the feasible set
    for j, (lb, ub) in enumerate(zip(*bounds)):
        X.data[..., j].clamp_(lb, ub)  # need to do this on the data not X itself

    # store the optimization trajecatory
    X_traj.append(X.detach().clone())
    loss_traj.append(losses.detach().clone())

    if (i + 1) % 10 == 0:
        print(f"Iteration {i+1:>3}/{TRAINING_ITERATIONS:>3} - Loss: {loss.item():>4.3f}")

print("Target composition : ", target_comp)
with torch.no_grad():
    loss_optim, spectra_optim = simulator(X_traj[-1].squeeze())
    print("Optimized composition : ", X_traj[-1].squeeze()[torch.argmin(loss_optim)])
    for i in range(NUM_RESTARTS):
        fig, ax = plt.subplots(figsize=(4, 4))
        mu = spectra_optim[i,:,0].cpu().squeeze().numpy()
        sigma = spectra_optim[i,:,1].cpu().squeeze().numpy()
        ax.plot(sim.t, target, label="Target")
        ax.plot(sim.t, mu, label="Best Estimate")
        ax.fill_between(sim.t, mu-sigma, mu+sigma,  
                        color='grey', alpha=0.5, label="Uncertainity"
                        )
        ax.set_title("Loss : %.2f"%loss_optim[i].item())
        ax.legend()
        plt.savefig(SAVE_DIR+"comparision_%d.png"%i)
        plt.close()

# Compute loss function on a grid for plotting
grid_comps = get_twod_grid(30, bounds=bounds.cpu().numpy())
grid_loss, grid_spectra = simulator(torch.from_numpy(grid_comps).to(device))

optim_result = {"X_traj" : torch.stack(X_traj, dim=1).squeeze(),
                "loss" : torch.stack(loss_traj, dim=1).squeeze(),
                "spectra" : spectra_optim,
                "target_y" : yt,
                "target_x" : xt,
                "grid_loss" : grid_loss,
                "grid_spectra" : grid_spectra,
                "grid_comps" : grid_comps
                }
torch.save(optim_result, SAVE_DIR+"optim_traj.pkl")