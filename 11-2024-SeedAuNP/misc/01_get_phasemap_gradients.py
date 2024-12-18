import torch
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import pandas as pd 
import pdb, argparse, json, glob, pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from torch.autograd import Variable
from activephasemap.models.np import NeuralProcess
from activephasemap.simulators import UVVisExperiment
from activephasemap.models.xgb import XGBoost
from activephasemap.utils import *

ITERATION = 14

DATA_DIR = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/11-2024-SeedAuNP/"
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap/activephasemap/pretrained/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]
TOTAL_ITERATIONS  = len(glob.glob(DATA_DIR+"/output/comp_model_*.json"))

# Load trained NP model for p(y|z)
np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
np_model.load_state_dict(torch.load(DATA_DIR+'/output/np_model_%d.pt'%(ITERATION), map_location=device, weights_only=True))

# Load trained composition to latent model for p(z|c)
xgb_model_args = {"objective": "reg:squarederror",
                  "max_depth": 3,
                  "eta": 0.1,
                  "eval_metric": "rmse"
                  }
comp_model = XGBoost(xgb_model_args)
comp_model.load(DATA_DIR+"/output/comp_model_%d.json"%ITERATION)

design_space_bounds = [(0.0, 35.0), (0.0, 35.0)]
wav = np.load(DATA_DIR+"/data/wav.npy")
t_np = (wav-min(wav))/(max(wav) - min(wav))
def get_spectrum(c):
    # predict z value from the comp model
    z_mu, z_std = comp_model.predict(c)   
    nr, nb, dz = z_mu.shape
    nz  = 20
    z_dist = torch.distributions.Normal(z_mu, z_std)
    z = z_dist.rsample(torch.Size([nz])).view(nz*nr*nb, dz)
    t = torch.from_numpy(t_np).repeat(nz*nr*nb, 1, 1).to(device)
    t = torch.swapaxes(t, 1, 2)
    y_samples, _ = np_model.xz_to_y(t, z)

    mu_pred = y_samples.view(nz, nr, nb, len(t_np), 1).mean(dim=0)

    return mu_pred.squeeze() 

def compute_gradient(x_samples, y_samples, f, epsilon=1e-2):
    """
    Approximates the gradient ∇f(x) using finite differences in PyTorch.
    :param x_samples: Input points in R^d (n_samples, n_dim) as a torch.Tensor.
    :param y_samples: Corresponding outputs in L2 space (n_samples, n_discretized) as a torch.Tensor.
    :param f: Function f(x) to compute the gradient.
    :param epsilon: Small step for finite differences.
    :return: Gradients of shape (n_samples, n_dim, n_discretized).
    """
    n_samples, _, n_dim = x_samples.shape
    _, n_discretized = y_samples.shape
    x_forward = x_samples.clone()
    h = 0.5*torch.norm(x_samples, dim=(1,2)).mean().item() + epsilon
    gradients = torch.zeros((n_samples, n_dim, n_discretized)) 
    for dim in range(n_dim):  # Loop over dimensions of x
        perturb = torch.zeros_like(x_samples)
        perturb[...,dim] = epsilon
        x_plus = x_samples + perturb
        x_minus = x_samples - perturb
        f_grad = (f(x_plus) - f(x_minus)) / (2 * h)
        
        gradients[:, dim, :] = f_grad 
        
    return gradients  # Shape: (n_samples, n_dim, n_discretized)

grid_comps = get_twod_grid(20, np.asarray(design_space_bounds).T)
n_samples, dx = grid_comps.shape
grid_comps = torch.from_numpy(grid_comps).to(device).view(n_samples, 1, dx)
grid_spectra = get_spectrum(grid_comps)
print("grid data : ", grid_comps.shape, grid_spectra.shape)
gradients = compute_gradient(grid_comps, grid_spectra, get_spectrum)

print("gradient of spectrum : ", gradients.shape)

with torch.no_grad():
    example_idx = 10
    plt.figure(figsize=(8, 6))
    plt.plot(t_np, grid_spectra[example_idx,:].cpu().numpy(), color="k")
    for dim, label in zip(range(2), ["x1", "x2"]):
        plt.plot(t_np, gradients[example_idx, dim, : ].cpu().numpy(), label=f"∂f/∂{label}")
    plt.title(f"Gradients in L2 Space for Sample Point {grid_comps[example_idx].cpu().numpy().squeeze()}")
    plt.xlabel("t (Discretized L2 space)")
    plt.ylabel("Gradient")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig("plot_function_gradient.png")
    plt.close()

    fig, ax = plt.subplots()
    norms =  torch.linalg.norm(gradients, dim=-1).cpu().numpy()
    print("gradient norms : ", norms.shape)
    ax.tricontourf(grid_comps[...,0].cpu().numpy().squeeze(), 
                   grid_comps[..., 1].cpu().numpy().squeeze(), 
                   np.linalg.norm(norms, axis=1)
                   )
    ax.quiver(grid_comps[...,0].cpu().numpy().squeeze(), 
              grid_comps[..., 1].cpu().numpy().squeeze(), 
              norms[:,0], 
              norms[:,1],
              color="w"
              )

    plt.savefig("gradients_quiver.png")
    plt.close()