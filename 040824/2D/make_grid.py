import numpy as np 
import matplotlib.pyplot as plt 
import time, os, traceback, shutil, warnings, pickle, pdb, glob, json
import ray 

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from botorch.utils.transforms import normalize

from activephasemap.models.np import NeuralProcess
from activephasemap.models.gp import MultiTaskGP
from activephasemap.utils.simulators import UVVisExperiment
from activephasemap.utils.settings import from_comp_to_spectrum, get_twod_grid, AutoPhaseMapDataSet

from scipy.stats import qmc

# Specify variables
DATA_DIR = "./"
SAVE_DIR = "./grid/"
ITERATION = len(glob.glob(DATA_DIR+"data/spectra_*.npy"))
print("Using data until %d iterations"%ITERATION)
design_space_bounds = [(0.0, 87.0), (0.0,11.0)]

if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]

expt = UVVisExperiment(design_space_bounds, ITERATION, DATA_DIR+"/data/")
expt.generate(use_spline=True)
gp_model_args = {"model":"gp", "num_epochs" : 1, "learning_rate" : 1e-3, "verbose": 1}

# Load trained GP model for p(z|c)
train_x = torch.load(DATA_DIR+'/output/train_x_%d.pt'%ITERATION, map_location=device)
train_y = torch.load(DATA_DIR+'/output/train_y_%d.pt'%ITERATION, map_location=device)
train_y_std = 0.1*torch.ones_like(train_y)
bounds = expt.bounds.to(device)
normalized_x = normalize(train_x, bounds).to(train_x)
gp_model = MultiTaskGP(normalized_x, train_y, gp_model_args, expt.dim, N_LATENT, train_y_std)
gp_state_dict = torch.load(DATA_DIR+'/output/gp_model_%d.pt'%(ITERATION), map_location=device)
gp_model.load_state_dict(gp_state_dict)

# Load trained NP model for p(y|z)
np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
np_model.load_state_dict(torch.load(DATA_DIR+'/output/np_model_%d.pt'%(ITERATION), map_location=device))

# Construct data on a grid
N_GRID_SAMPLES = 2000
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=N_GRID_SAMPLES)

x = design_space_bounds[0][0] + sample[:,0]*(design_space_bounds[0][1] - design_space_bounds[0][0])
y = design_space_bounds[1][0] + sample[:,1]*(design_space_bounds[1][1] - design_space_bounds[1][0])
grid_comps = np.vstack((x,y)).T

# grid_comps = get_twod_grid(25, bounds=expt.bounds.cpu().numpy())
# N_GRID_SAMPLES = grid_comps.shape[0] 

fig, ax = plt.subplots()
ax.scatter(grid_comps[:,0], grid_comps[:,1])
plt.savefig(SAVE_DIR + "samples.png")
plt.close()

n_spectra_dim =  expt.t.shape[0]
grid_spectra = np.zeros((N_GRID_SAMPLES, n_spectra_dim))
with torch.no_grad():
    for i in range(N_GRID_SAMPLES):
        mu, _ = from_comp_to_spectrum(expt, 
                                      gp_model,
                                      np_model, 
                                      grid_comps[i,:].reshape(1, -1)
                                      )
        grid_spectra[i,:] = mu.cpu().squeeze().numpy()

np.save(SAVE_DIR+"grid_spectra.npy", grid_spectra)
np.save(SAVE_DIR+"grid_comps.npy", grid_comps)

for i in range(3):
    fig, ax = plt.subplots()
    rids = np.random.randint(N_GRID_SAMPLES, size=20)
    for r in rids:
        ax.plot(expt.t, grid_spectra[r,:])
    plt.savefig(SAVE_DIR + "spectra_%d.png"%i)
    plt.close()