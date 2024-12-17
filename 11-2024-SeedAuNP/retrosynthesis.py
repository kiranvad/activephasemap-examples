import os, sys, shutil, pdb, argparse,json, glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from botorch.utils.sampling import draw_sobol_samples
torch.set_default_dtype(torch.double)

from activephasemap.models.np import NeuralProcess
from activephasemap.simulators import UVVisExperiment
from activephasemap.models.xgb import XGBoost
from activephasemap.utils import *
from funcshape.functions import Function, SRSF
import optimum_reparamN2 as orN2

TRAINING_ITERATIONS = 500 # total iterations for each optimization
NUM_RESTARTS = 4 # number of optimization from random restarts
LEARNING_RATE = 1e-1
TARGET_SHAPE_ID = 1 # chose from [0 - "sphere", 1 - "nanorod"]

TARGET_SHAPES = ["sphere", "nanorod"]
print("Retrosynthesizing %s"%TARGET_SHAPES[TARGET_SHAPE_ID])
SAVE_DIR = "./retrosynthesis/%s/"%TARGET_SHAPES[TARGET_SHAPE_ID]
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)

DATA_DIR = './output/'
ITERATION = len(glob.glob(DATA_DIR+"comp_model_*.json"))
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]

# Load trained NP model for p(y|z)
np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
np_model.load_state_dict(torch.load(DATA_DIR+'np_model_%d.pt'%(ITERATION), map_location=device, weights_only=True))
np_model.train(False)

# Load trained composition to latent model for p(z|c)
comp_model = XGBoost(xgb_model_args)
comp_model.load(DATA_DIR+"comp_model_%d.json"%ITERATION)

design_space_bounds = [(0.0, 35.0), (0.0, 35.0)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)
DESIGN_SPACE_DIM = len(design_space_bounds)

# Create a target spectrum
TARGETS_DIR = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/11-2024-SeedAuNP/retrosynthesis/"
if TARGET_SHAPE_ID==0:
    target = np.load(TARGETS_DIR+"target_sphere.npz")
else:
    target = np.load(TARGETS_DIR+"target_nanorod.npz")
wav = target["x"]
n_domain = len(wav)
t = (wav-min(wav))/(max(wav)-min(wav))
xt = torch.from_numpy(t).to(device)
yt = torch.from_numpy(target["y"]).to(device)

def min_max_normalize(x):
    min_x = x.min(dim=1).values 
    max_x = x.max(dim=1).values
    x_norm = (x - min_x[:,None])/((max_x-min_x)[:,None])
    
    return x_norm

class Simulator(torch.nn.Module):
    def __init__(self, xt, c2z, z2y, nz=128):
        super().__init__()
        self.c_to_z = c2z 
        self.z_to_y = z2y
        self.t = xt
        self.nz = nz

    def forward(self, x):
        """
        x - should be of shape (n, m, dx)
        """
        z_mu, z_std = self.c_to_z.predict(x)
        nr, nb, dz = z_mu.shape
        z_dist = torch.distributions.Normal(z_mu, z_std)
        z = z_dist.rsample(torch.Size([self.nz])).view(self.nz*nr*nb, dz)
        time = self.t.repeat(self.nz*nr*nb, 1, 1).to(device)
        time = torch.swapaxes(time, 1, 2)
        y_samples, _ = self.z_to_y.xz_to_y(time, z)

        mu = y_samples.view(self.nz, nr, nb, len(self.t), 1).mean(dim=0).squeeze()
        sigma = y_samples.view(self.nz, nr, nb, len(self.t), 1).std(dim=0).squeeze()

        return torch.stack((mu, sigma), dim=-1)


def amplitude_phase_distance(t_np, f1, f2, **kwargs):
    t_tensor = torch.tensor(t_np, dtype=f1.dtype, device=f2.device)
    f1_ = Function(t_tensor, f1.reshape(-1,1))
    f2_ = Function(t_tensor, f2.reshape(-1,1))
    q1, q2 = SRSF(f1_), SRSF(f2_)

    delta = q1.qx-q2.qx
    if (delta==0).all():
        amplitude, phase = 0.0, 0.0
    else:
        q1_np = q1.qx.clone().detach().cpu().squeeze().numpy()
        q2_np = q2.qx.clone().detach().cpu().squeeze().numpy()
        
        gamma = orN2.coptimum_reparam(np.ascontiguousarray(q1_np), 
                                      t_np,
                                      np.ascontiguousarray(q2_np), 
                                      kwargs.get("lambda", 0.0),
                                      kwargs.get("grid_dim", 7)
                                    )
        gamma = (t_np[-1] - t_np[0]) * gamma + t_np[0]
    gamma_tensor = torch.tensor(gamma, dtype=f1.dtype, device=f1.device)
    warping = Function(t_tensor.squeeze(), gamma_tensor.reshape(-1,1))

    # Compute amplitude
    gam_dev = torch.abs(warping.derivative(warping.x))
    q_gamma = q2(warping.fx)
    y = (q1.qx.squeeze() - (q_gamma.squeeze() * torch.sqrt(gam_dev).squeeze())) ** 2
    integral = torch.trapezoid(y, q1.x)
    amplitude = torch.sqrt(integral)

    # Compute phase
    theta = torch.trapezoid(torch.sqrt(gam_dev).squeeze(), x=warping.x)
    phase = torch.arccos(torch.clamp(theta, -1, 1))
    if amplitude.isnan() or phase.isnan():
        pdb.set_trace()
    return amplitude, phase

def mse_loss(y_pred):
    num_points, _ = y_pred.shape
    target = yt.repeat(num_points, 1)
    target_ = min_max_normalize(target)
    mu_ = min_max_normalize(y_pred)

    loss = ((target_-mu_)**2).sum(dim=1)

    return loss.sum(), torch.tensor(loss, dtype=y_pred.dtype, device=y_pred.device)   

def ap_loss(y_pred):
    alpha = 0.5
    num_points, _ = y_pred.shape
    target = yt.repeat(num_points, 1)
    target_ = min_max_normalize(target)
    mu_ = min_max_normalize(y_pred)
    loss = 0.0
    loss_values = []
    for i in range(num_points):
        amplitude, phase = amplitude_phase_distance(t, mu_[i,:], target_[i,:])
        dist = (1-alpha)*amplitude + (alpha)*phase
        loss += dist 
        loss_values.append(dist.item())
        loss += dist
    
    return loss, torch.tensor(loss_values, dtype=y_pred.dtype, device=y_pred.device)

def closure():
    global loss_values
    global loss
    global spectra

    lbfgs.zero_grad()
    spectra = sim(X)
    loss, loss_values = loss_fn(spectra[...,0]) 
    loss.backward()

    return loss

sim = Simulator(xt, comp_model, np_model).to(device)

# Initialize using random Sobol sequence sampling
X = draw_sobol_samples(bounds=bounds, n=NUM_RESTARTS, q=1).to(device)
X.requires_grad_(True)

lbfgs = torch.optim.LBFGS([X],
                    history_size=10, 
                    max_iter=4, 
                    line_search_fn="strong_wolfe")

X_traj, loss_traj, spectra_traj = [], [], []
loss_fn = ap_loss
# run a basic optimization loop
for i in range(TRAINING_ITERATIONS):
    lbfgs.step(closure)
    
    # clamp values to the feasible set
    for j, (lb, ub) in enumerate(zip(*bounds)):
        X.data[..., j].clamp_(lb, ub) 

    # store the optimization trajectory
    # clone and detaching is importat to not meddle with the autograd
    X_traj.append(X.clone().detach())
    loss_traj.append(loss_values.clone().detach())
    spectra_traj.append(spectra.clone().detach())
    if (i + 1) % 1 == 0:
        print(f"Iteration {i+1:>3}/{TRAINING_ITERATIONS:>3} - Loss: {loss.item():>4.3f}; dX: {X.grad.mean():>.2e}")

# Compute loss function on a grid for plotting
with torch.no_grad():
    grid_comps = get_twod_grid(15, bounds=bounds.cpu().numpy())
    grid_spectra = sim(torch.from_numpy(grid_comps).view(225, 1, 2).to(device))
    _, grid_loss = loss_fn(grid_spectra[...,0])
    print(grid_loss.shape)

with torch.no_grad():
    spectra_optim = sim(X_traj[-1])
    _, loss_optim = loss_fn(spectra_optim[...,0])
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
        axs[1].tricontourf(grid_comps[:,0], grid_comps[:,1], grid_loss.detach().cpu().numpy(), cmap="binary")
        axs[1].plot(traj[:,0], traj[:,1],
                    lw=2,c='tab:red', label="Trajectory"
                    )
        axs[1].scatter(traj[0,0], traj[0,1],
                       s=100,c='tab:red',marker='.',
                       zorder=10,lw=2, label="Initial"
                       )
        axs[1].scatter(traj[-1,0], traj[-1,1],
                       s=100,c='tab:red',marker='+',
                       zorder=10,lw=2, label="Final"
                       )
        axs[1].set_xlim(*design_space_bounds[0])
        axs[1].set_ylim(*design_space_bounds[1])
        axs[1].legend()
        plt.savefig(SAVE_DIR+"comparision_%d.png"%i)
        plt.close()

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