import torch
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.double)

import os, pdb, sys, shutil
sys.path.append('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/11-2024-SeedAuNP/misc/mie_scattering')
from mie import *

from activephasemap.simulators import UVVisExperiment
from scipy.signal import find_peaks

SAVE_DIR = "./results/"
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)

DATA_DIR = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/11-2024-SeedAuNP"
ITERATION = 14
grid_data = np.load(DATA_DIR + "/paper/grid_data_10_%d.npz"%ITERATION)
grid_spectra = grid_data["spectra"]
design_space_bounds = [(0.0, 35.0), (0.0, 35.0)]
expt = UVVisExperiment(design_space_bounds, DATA_DIR+"/data/")
expt.read_iter_data(ITERATION)
expt.generate(use_spline=True)
wavelengths = torch.from_numpy(expt.wl)

def normalize(x):
    return (x-min(x))/(1e-3+ max(x) - min(x))

def featurize(x,y):
    "Use peak locations to determine morphology"
    peaks, _ = find_peaks(y, prominence=0.01, width=0.3)
    if len(peaks)==0:
        shape = 0
    elif (x[peaks]>600).any():
        shape = 2
    elif not (x[peaks]>600).any():
        shape = 1
        
    return shape

def objective_mixed(target_spectra, x):
    error = 0.0
    s_sphere = torch.zeros_like(wavelengths)
    s_nanorod = torch.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        s_sphere[i] = sphere_extinction(wl, *x[:2])
        s_nanorod[i] = nanorod_extinction(wl, *x[2:-1])
    
    s_query = x[-1]*normalize(s_sphere)+(1-x[-1])*normalize(s_nanorod)
    
    error = (s_query-normalize(target_spectra))**2 
           
    return error.sum()

def objective_sphere(target_spectra, x):
    error = 0.0
    s_query = torch.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        s_query[i] = sphere_extinction(wl, *x)
    
    error = (normalize(s_query)-normalize(target_spectra))**2 
           
    return error.sum()


parameters_bounds_mixed = [(6.0, 50.0), # sphere radius (mu)
                       (1.0, 10), # dieletric constant for sphereical medium
                       (1.1, 5.0), # nanorod aspect ratio
                       (1.0, 10.0), # dieletric constant for nanorod medium
                       (0.0, 1.0), # mixed model weights
                       ]

parameters_bounds_sphere = [(2.0, 10.0), # sphere radius (mu)
                       (1.0, 10), # dieletric constant for sphereical medium
                       ]

def get_fit_params(spectra):
    feats = featurize(expt.wl, spectra)
    if feats==1:
        objective = lambda x : objective_sphere(spectra, x)
        parameters_bounds = parameters_bounds_sphere
    elif feats==2:
        objective = lambda x : objective_mixed(spectra, x)
        parameters_bounds = parameters_bounds_mixed    
    else:
        warnings.warn("Seed solution spectra need not be.")
        objective = None 
        parameters_bounds = None 

    return feats, objective, parameters_bounds 

target_spectra = torch.from_numpy(expt.spectra_normalized[0,:])
feats, objective, parameters_bounds = get_fit_params(target_spectra)
fit_kwargs = {"n_iterations": 100, "n_restarts": 10, "epsilon": 0.2, "lr":0.01}
best_X, best_error = fit_mie_scattering(objective, parameters_bounds, **fit_kwargs)

if feats==2:
    s_sphere = torch.zeros_like(wavelengths)
    s_nanorod = torch.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        s_sphere[i] = sphere_extinction(wl, *best_X[:2])
        s_nanorod[i] = nanorod_extinction(wl, *best_X[2:-1])
    spectra_mixture_optimized = best_X[-1]*normalize(s_sphere) + (1-best_X[-1])*normalize(s_nanorod)

elif feats==1:
    spectra_mixture_optimized = torch.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        spectra_mixture_optimized[i] = sphere_extinction(wl, *best_X)
    spectra_mixture_optimized = normalize(spectra_mixture_optimized)

np.savez(SAVE_DIR+"res.npz", 
         best_X = best_X.numpy(), 
         best_error = best_error,
         target = normalize(target_spectra.detach().numpy()),
         optimized = spectra_mixture_optimized.detach().numpy()
         )

fig, ax = plt.subplots()
ax.plot(wavelengths, 
        normalize(target_spectra.detach().numpy()), 
        color="k", 
        label="Target"
        )
ax.plot(wavelengths, 
        spectra_mixture_optimized.detach().numpy(), 
        color="k", 
        ls="--",
        label="Optimized"
        )
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Extinction Coefficient")
ax.set_title("Error : %.2f"%best_error)
ax.legend()
plt.savefig(SAVE_DIR + "optimization_comparision.png")
plt.close()