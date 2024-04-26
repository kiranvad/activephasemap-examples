import numpy as np 
import matplotlib.pyplot as plt 
import time, os, traceback, shutil, warnings, pickle, pdb
import ray 

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from botorch.utils.transforms import normalize

from activephasemap.np.neural_process import NeuralProcess
from activephasemap.utils.settings import initialize_model
from activephasemap.test_functions.phasemaps import ExperimentalTestFunction
from activephasemap.utils.simulators import UVVisExperiment
from activephasemap.utils.settings import from_comp_to_spectrum, get_twod_grid, AutoPhaseMapDataSet

from scipy.stats import qmc

# Specify variables
N_GRID_SAMPLES = 1000 

SAVE_DIR = "./grid/"
ITERATION = 3
DATA_DIR = "./"
design_space_bounds = [(0.0, 87.0), (0.0,11.0)]

if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

expt = UVVisExperiment(ITERATION, DATA_DIR+"/data/")
expt.generate()
test_function = ExperimentalTestFunction(sim=expt, bounds=design_space_bounds)
gp_model_args = {"model":"gp", "num_epochs" : 1, "learning_rate" : 1e-3, "verbose": 1}
np_model_args = {"num_iterations": 1, "verbose":True, "print_freq":100, "lr":5e-4}
input_dim = test_function.dim
output_dim = 2 

# Load trained GP model for p(z|c)
gp_model = initialize_model(gp_model_args, input_dim, output_dim, device)
train_x = torch.load(DATA_DIR+'/output/train_x_%d.pt'%ITERATION, map_location=device)
train_y = torch.load(DATA_DIR+'/output/train_y_%d.pt'%ITERATION, map_location=device)
bounds = test_function.bounds.to(device)
normalized_x = normalize(train_x, bounds).to(train_x)
gp_state_dict = torch.load(DATA_DIR+'/output/gp_model_%d.pt'%(ITERATION), map_location=device)
loss = gp_model.fit(normalized_x, train_y)
gp_model.load_state_dict(gp_state_dict)

# Load trained NP model for p(y|z)
np_model = NeuralProcess(1, 1, 128, 2, 128).to(device)
np_model.load_state_dict(torch.load(DATA_DIR+'/output/np_model_%d.pt'%(ITERATION), map_location=device))

# Construct data on a grid
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=N_GRID_SAMPLES)

x = design_space_bounds[0][0] + sample[:,0]*(design_space_bounds[0][1] - design_space_bounds[0][0])
y = design_space_bounds[1][0] + sample[:,1]*(design_space_bounds[1][1] - design_space_bounds[1][0])
grid_comps = np.vstack((x,y)).T

fig, ax = plt.subplots()
ax.scatter(grid_comps[:,0], grid_comps[:,1])
plt.savefig(SAVE_DIR + "samples.png")
plt.close()

n_spectra_dim =  test_function.sim.t.shape[0]
grid_spectra = np.zeros((N_GRID_SAMPLES, n_spectra_dim))
with torch.no_grad():
    for i in range(N_GRID_SAMPLES):
        mu, _ = from_comp_to_spectrum(test_function, 
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
        ax.plot(test_function.sim.t, grid_spectra[r,:])
    plt.savefig(SAVE_DIR + "spectra_%d.png"%i)
    plt.close()