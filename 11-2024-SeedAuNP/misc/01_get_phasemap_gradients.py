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
from funcshape.functions import Function

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
    nz  = 32
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

    return tangent_vectors, grad, grad_norm

def evaluate_batch_gradients(x, f):
    n_samples, dx = x.shape
    batch_comps = torch.from_numpy(grid_comps).to(device).view(n_samples, 1, dx)
    batch_spectra = f(batch_comps)
    grad_norms = torch.zeros(n_samples)
    tangent_vectors = torch.zeros((n_samples, len(t_np), dx))
    grad = torch.zeros(n_samples, dx)
    norm = lambda fx, tv : fisher_rao_norm(torch.from_numpy(t_np).to(device), fx, tv)
    for i in range(n_samples):
        xi = batch_comps[i,...].view(1, 1, dx).clone().detach().requires_grad_(True)
        tangent_vectors_xi, grads_xi, grad_norms_xi = compute_gradient(xi, f, norm)
        grad_norms[i] = grad_norms_xi
        tangent_vectors[i,...] = tangent_vectors_xi 
        grad[i,...] = grads_xi
    
    return batch_spectra, grad_norms, tangent_vectors, grad

def adaptive_sampling(p, points, refinement_threshold=0.8, max_iterations=4):
    """
    Refines the grid adaptively based on the density function p(x).
    """
    for _ in range(max_iterations):
        # Evaluate p(x) on current points
        p_values,_,_ = p(points)
        high_density = points[p_values > refinement_threshold*max(p_values)]
        
        # Subdivide regions with high density
        subdivided = []
        for point in high_density:
            step = (points[1,0] - points[0,0]) / 2  # Half the step size
            new_points = [point + np.array([dx, dy]) * step 
                          for dx in [-0.5, 0.5] for dy in [-0.5, 0.5]]
            subdivided.extend(new_points)
        
        # Add new points and ensure uniqueness
        points = np.vstack([points, subdivided])
        points = np.unique(points, axis=0)
    
    return points

# compute values on a grid
grid_comps = get_twod_grid(20, np.asarray(design_space_bounds).T)
grid_spectra, grad_norms_grid, tangent_vectors_grid, grad_grid = evaluate_batch_gradients(grid_comps, get_spectrum)

print("grid data : ", grid_comps.shape, grid_spectra.shape)
print("Gradient data : ", grad_norms_grid.shape, tangent_vectors_grid.shape, grad_grid.shape)

# # perform adaptive sampling
# grid_comps = adaptive_sampling(grad_norms_grid.detach().cpu().squeeze(),
#                                grid_comps.cpu().squeeze()
#                             )
# n_samples, dx = grid_comps.shape
# grid_comps = torch.from_numpy(grid_comps).to(device).view(n_samples, 1, dx)
# grid_spectra = get_spectrum(grid_comps)
# grad_norms_grid, tangent_vectors_grid, grad_grid = evaluate_grid_gradients(grid_comps, grid_spectra)
# print("Gradient data : ", grad_norms_grid.shape, tangent_vectors_grid.shape, grad_grid.shape)

with torch.no_grad():
    example_idx = np.random.randint(0, grid_comps.shape[0]-1)
    plt.figure(figsize=(8, 6))
    plt.plot(t_np, grid_spectra[example_idx,:].cpu().numpy(), color="k")
    for dim, label in zip(range(2), ["x1", "x2"]):
        plt.plot(t_np, tangent_vectors_grid[example_idx, :, dim ].cpu().numpy(), label=f"∂f/∂{label}")
    plt.title(f"Gradients in L2 Space for Sample Point {grid_comps[example_idx, :]}")
    plt.xlabel("t (Discretized L2 space)")
    plt.ylabel("Gradient")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig("plot_function_gradient.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.tricontourf(grid_comps[...,0], 
                   grid_comps[..., 1], 
                   grad_norms_grid.detach().cpu().squeeze(),
                   levels=50
                   )
    ax.quiver(grid_comps[...,0], 
              grid_comps[..., 1], 
              grad_grid[:,0].detach().cpu().squeeze(), 
              grad_grid[:,1].detach().cpu().squeeze(),
              color="w"
              )

    plt.savefig("gradients_quiver.png")
    plt.close()