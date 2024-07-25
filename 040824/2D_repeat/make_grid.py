import numpy as np 
import matplotlib.pyplot as plt 
import time, os, traceback, shutil, warnings, pickle, pdb, glob, json
import ray 

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from botorch.utils.transforms import normalize

from activephasemap.models.np import NeuralProcess
from activephasemap.models.mlp import MLP
from activephasemap.utils.simulators import UVVisExperiment
from activephasemap.utils.visuals import  get_twod_grid

from scipy.stats import qmc
from tqdm import tqdm

# Specify variables
DATA_DIR = "./"
SAVE_DIR = "./grid/"
ITERATION = 8
design_space_bounds = [(0.0, 87.0), (0.0,11.0)]

if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]

def from_comp_to_spectrum(t, c, comp_model, np_model):
    ci = torch.tensor(c).to(device)
    z_mu, z_std = comp_model.mlp(ci)
    z_dist = torch.distributions.Normal(z_mu, z_std)
    z = z_dist.sample(torch.Size([100]))
    t = torch.from_numpy(expt.t).repeat(100, 1, 1).to(device)
    t = torch.swapaxes(t, 1, 2)
    y_samples, _ = np_model.xz_to_y(t, z)

    mean_pred = y_samples.mean(dim=0, keepdim=True)
    sigma_pred = y_samples.std(dim=0, keepdim=True)
    mu_ = mean_pred.cpu().squeeze()
    sigma_ = sigma_pred.cpu().squeeze() 

    return mu_, sigma_  

class UVVisPlateData(UVVisExperiment):
    def __init__(self, bounds, direc):
        super().__init__(bounds, direc)
        self.comps = np.load(direc+"comps_train.npy").astype(np.double)
        self.spectra = np.load(direc+"spectra_train.npy")
        self.wav = np.load(self.dir+'wav.npy')

# Set up a synthetic data emulating an experiment
expt = UVVisPlateData(design_space_bounds, DATA_DIR)
expt.generate(use_spline=True)
bounds = expt.bounds.to(device)

# Load trained MLP model for p(z|c)
train_x = torch.load(DATA_DIR+'/train_x_%d.pt'%ITERATION, map_location=device)
train_z_mean = torch.load(DATA_DIR+'/train_z_mean_%d.pt'%ITERATION, map_location=device)
train_z_std = torch.load(DATA_DIR+'/train_z_std_%d.pt'%ITERATION, map_location=device)
mlp = MLP(train_x, train_z_mean, train_z_std)
mlp_state_dict = torch.load(DATA_DIR+'/comp_model_%d.pt'%(ITERATION), map_location=device)
mlp.load_state_dict(mlp_state_dict)

# Load trained NP model for p(y|z)
NP = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
NP.load_state_dict(torch.load(DATA_DIR+'/np_model_%d.pt'%(ITERATION), map_location=device))

# Construct data on a grid
N_GRID_SAMPLES = 2000
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=N_GRID_SAMPLES)

x = design_space_bounds[0][0] + sample[:,0]*(design_space_bounds[0][1] - design_space_bounds[0][0])
y = design_space_bounds[1][0] + sample[:,1]*(design_space_bounds[1][1] - design_space_bounds[1][0])
grid_comps = np.vstack((x,y)).T

fig, ax = plt.subplots()
ax.scatter(grid_comps[:,0], grid_comps[:,1])
plt.savefig(SAVE_DIR + "samples.png")
plt.close()

n_spectra_dim =  expt.t.shape[0]
grid_spectra = np.zeros((N_GRID_SAMPLES, n_spectra_dim))
with torch.no_grad():
    for i in tqdm(range(N_GRID_SAMPLES)):
        mu, _ = from_comp_to_spectrum(expt.t, 
                                      grid_comps[i,:],
                                      mlp,
                                      NP
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