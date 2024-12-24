import torch
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import pandas as pd 
import pdb, argparse, json, glob, pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from torch.profiler import profile, record_function, ProfilerActivity

from activephasemap.models.np import NeuralProcess
from activephasemap.simulators import UVVisExperiment
from activephasemap.models.xgb import XGBoost
from activephasemap.utils import *
from funcshape.functions import Function
import adaptive
from scipy.spatial import Delaunay

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
    nz  = 128
    z_dist = torch.distributions.Normal(z_mu, z_std)
    z = z_dist.rsample(torch.Size([nz])).view(nz*nr*nb, dz)
    t = torch.from_numpy(t_np).repeat(nz*nr*nb, 1, 1).to(device)
    t = torch.swapaxes(t, 1, 2)
    y_samples, _ = np_model.xz_to_y(t, z)

    mu_pred = y_samples.view(nz, nr, nb, len(t_np), 1).mean(dim=0)

    return mu_pred.squeeze() 

def l2_norm(x, fx, tangent_vectors):
    grad = torch.trapezoid(tangent_vectors**2, x=x, dim=0)
    grad_norm = grad.sum()

    return grad, grad_norm 

def fisher_rao_norm(x, y, tv):
    ft = Function(x, y.reshape(-1,1))
    vt_x1 = Function(x, tv[:,0].reshape(-1,1))
    vt_x2 = Function(x, tv[:,1].reshape(-1,1))
    scale = torch.sqrt(torch.abs(ft.derivative(x)) + 1e-3)
    srsf = torch.stack((vt_x1.derivative(x)/scale , vt_x2.derivative(x)/scale)).squeeze().T

    grad = 0.25*torch.trapezoid(srsf**2, x=x, dim=0)
    grad_norm = grad.sum()

    return grad, grad_norm  

def compute_gradient(xt, f, norm):
    yt = f(xt).squeeze()
    df_dx = []
    for i in range(len(t_np)): 
        yt[i].backward(retain_graph=True) 
        df_dx.append(xt.grad.clone())
        xt.grad.zero_()

    df_dx = torch.stack(df_dx).squeeze()
    tangent_vectors = xt.squeeze()*df_dx
    grad, grad_norm = norm(yt, tangent_vectors.squeeze())

    del yt, df_dx, tangent_vectors
    torch.cuda.empty_cache()  # Clear the GPU cache

    return grad, grad_norm

def evaluate_batch_gradients(x, f):
    n_samples, dx = x.shape
    batch_comps = torch.from_numpy(x).to(device)
    grad_norms, grad = [], []
    norm = lambda fx, tv : fisher_rao_norm(torch.from_numpy(t_np).to(device), fx, tv)
    for i in range(n_samples):
        print("Computing gradient of %d/%d"%(i, n_samples), end="\r", flush=True)
        xi = batch_comps[i,...].view(1, 1, dx).clone().detach().requires_grad_(True)
        grads_xi, grad_norms_xi = compute_gradient(xi, f, norm)
        grad_norms.append(grad_norms_xi)
        grad.append(grads_xi)
    
    return torch.stack(grad_norms), torch.stack(grad)

# compute values on a grid
grid_comps = get_twod_grid(18, np.asarray(design_space_bounds).T)
grad_norms_grid, grad_grid = evaluate_batch_gradients(grid_comps, get_spectrum)

print("grid data : ", grid_comps.shape)
print("Grid Gradient data : ", grad_norms_grid.shape, grad_grid.shape)

with torch.no_grad():
    fig, ax = plt.subplots()
    ctr = ax.tricontourf(grid_comps[...,0], grid_comps[..., 1], 
                   grad_norms_grid.detach().cpu().squeeze(),
                   levels=50
                   )
    plt.colorbar(ctr, label="Gradient Norm")
    ax.quiver(grid_comps[...,0], grid_comps[..., 1], 
              grad_grid[:,0].detach().cpu().squeeze(), 
              grad_grid[:,1].detach().cpu().squeeze(),
              color="w"
              )
    ax.scatter(grid_comps[...,0], grid_comps[..., 1], color="k", s=10)

    plt.savefig("gradients_quiver.png")
    plt.close()

"""Perform Adaptive Sampling of the PhaseMap gradients"""

def phasemap_gradient(xy):
    """A function to compute gradient at any given composition.

    xy : a tuple corresponding to a 2D composition.
    """
    c1, c2 = xy
    comp_np = np.array([c1, c2])
    # for some reason only the following can track the gradient of x
    # not directly creating the tensor using torch.tensor
    comp_tensor = torch.from_numpy(comp_np).to(device).view(1, 1, 2)
    x = comp_tensor.clone().detach().requires_grad_(True)

    norm = lambda fx, tv : fisher_rao_norm(torch.from_numpy(t_np).to(device), fx, tv)
    value,_ = compute_gradient(x, get_spectrum, norm)

    return value.detach().cpu().squeeze().numpy()

learner = adaptive.Learner2D(phasemap_gradient, design_space_bounds)
adaptive.runner.simple(learner, goal=lambda l: l.npoints > 100)
data = learner.data
with torch.no_grad():
    x, y = zip(*data.keys())
    tri = Delaunay(np.column_stack((x, y)))

    fig, ax = plt.subplots()
    contour =  ax.tricontourf(grid_comps[...,0],
                            grid_comps[..., 1], 
                            grad_norms_grid.detach().cpu().squeeze(),
                            levels=50
                            )
    plt.colorbar(contour, label="Gradient Norm")
    ax.scatter(x, y, s=15, color="k")
    for simplex in tri.simplices:
        pts = tri.points[simplex]
        ax.plot(pts[:, 0], pts[:, 1], 'k-')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig("adaptive_sampling.png")
    plt.close()