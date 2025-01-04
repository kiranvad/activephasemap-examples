import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.double)

import numpy as np
import matplotlib.pyplot as plt

import os, pdb, sys, shutil
sys.path.append('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/11-2024-SeedAuNP/misc/mie_scattering')
from mie import *

from activephasemap.simulators import UVVisExperiment
from scipy.signal import find_peaks

TESTING = True

if not TESTING:
    SAVE_DIR = "./results_sph/"
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

def gaussian_filter(input_tensor, sigma=1.0, truncate=4.0):
    """
    Apply a Gaussian filter to a 1D PyTorch tensor.

    Parameters:
    - input_tensor: torch.Tensor, 1D input tensor.
    - sigma: float, standard deviation for the Gaussian kernel.
    - truncate: float, truncate the kernel at this many standard deviations.

    Returns:
    - torch.Tensor, filtered tensor.
    """
    # Create the Gaussian kernel
    radius = int(truncate * sigma + 0.5)
    coords = torch.arange(-radius, radius + 1)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()  # Normalize the kernel

    # Reshape kernel for 1D convolution
    kernel = kernel.view(1, 1, -1)

    # Add batch and channel dimensions to the input tensor
    input_tensor = input_tensor.view(1, 1, -1)

    # Pad the input tensor
    pad_width = (radius, radius)
    input_tensor = F.pad(input_tensor, pad_width, mode='reflect')

    # Perform the convolution
    filtered_tensor = F.conv1d(input_tensor, kernel).squeeze()

    return filtered_tensor

def objective_mixed(target_spectra, x):
    error = 0.0
    s_sphere = torch.zeros_like(wavelengths)
    s_nanorod = torch.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        s_sphere[i] = sphere_extinction(wl, *x[:3])
        s_nanorod[i] = nanorod_extinction(wl, *x[3:-1])

    s_nanorod = gaussian_filter(s_nanorod)
    s_sphere = gaussian_filter(s_sphere)
    s_query = x[-1]*normalize(s_sphere)+(1-x[-1])*normalize(s_nanorod)
    
    error = (s_query-normalize(target_spectra))**2 
           
    return error.sum()

def objective_sphere(target_spectra, x):
    error = 0.0
    s_query = torch.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        s_query[i] = sphere_extinction(wl, *x)

    s_query = gaussian_filter(s_query)
    error = (normalize(s_query)-normalize(target_spectra))**2 
           
    return error.sum()


parameters_bounds_mixed = [(2.0, 50.0), # sphere radius (mu)
                        (0.001, 0.4), # sphere radius (sigma)
                       (1.0, 10), # dieletric constant for sphereical medium
                       (1.1, 5.0), # nanorod aspect ratio (mu)
                       (0.001, 0.4), # nanorod aspect ratio (sigma)
                       (1.0, 10.0), # dieletric constant for nanorod medium
                       (0.0, 1.0), # mixed model weights
                       ]

parameters_bounds_sphere = [(2.0, 50.0), # sphere radius (mu)
                        (0.001, 0.4), # sphere radius (sigma)
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

for sample_id in range(expt.spectra_normalized.shape[0]):
    if TESTING:
        sample_id = 75
    target_spectra = torch.from_numpy(expt.spectra_normalized[sample_id,:])
    feats, objective, parameters_bounds = get_fit_params(target_spectra)
    if feats in [0]:
        print("Skipping %d at composition : "%sample_id, expt.comps[sample_id,:])
        continue

    print("Fitting %d at composition : "%sample_id, expt.comps[sample_id,:], " with features %d"%feats)
    fit_kwargs = {"n_iterations": 1000, "n_restarts": 100, "epsilon": 0.5, "lr":0.01}
    best_X, best_error = fit_mie_scattering(objective, parameters_bounds, **fit_kwargs)

    if feats==2:
        s_sphere = torch.zeros_like(wavelengths)
        s_nanorod = torch.zeros_like(wavelengths)
        for i, wl in enumerate(wavelengths):
            s_sphere[i] = sphere_extinction(wl, *best_X[:3])
            s_nanorod[i] = nanorod_extinction(wl, *best_X[3:-1])

        s_nanorod = gaussian_filter(s_nanorod)
        s_sphere = gaussian_filter(s_sphere)

        spectra_optimized = best_X[-1]*normalize(s_sphere) + (1-best_X[-1])*normalize(s_nanorod)

    elif feats==1:
        s_sphere = torch.zeros_like(wavelengths)
        for i, wl in enumerate(wavelengths):
            s_sphere[i] = sphere_extinction(wl, *best_X)
        s_sphere = gaussian_filter(s_sphere)

        spectra_optimized = normalize(s_sphere)

    if not TESTING:
        np.savez(SAVE_DIR+"res_%d.npz"%sample_id,
                feats = feats,
                best_X = best_X.numpy(), 
                best_error = best_error,
                target = normalize(target_spectra.detach().numpy()),
                optimized = spectra_optimized.detach().numpy()
                )

    fig, ax = plt.subplots()
    ax.plot(wavelengths, 
            normalize(target_spectra.detach().numpy()), 
            color="k", 
            label="Target"
            )
    ax.plot(wavelengths, 
            spectra_optimized.detach().numpy(), 
            color="k", 
            ls="--",
            label="Optimized"
            )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Extinction Coefficient")
    ax.set_title("Error : %.2f"%best_error)
    ax.legend()
    if TESTING:
        plt.savefig("fit.png")
        plt.close()
        break
    else:
        plt.savefig(SAVE_DIR + "fit_%d.png"%sample_id)
        plt.close()