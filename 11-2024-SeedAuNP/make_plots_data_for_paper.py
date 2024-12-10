import torch
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import pandas as pd 
import pdb, argparse, json, glob, pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

from activephasemap.models.np import NeuralProcess
from activephasemap.simulators import UVVisExperiment
from activephasemap.models.xgb import XGBoost
from activephasemap.utils import *

parser = argparse.ArgumentParser(
                    prog='Train emulator of gold nanoparticle synthesis',
                    description='Perform a single iteration of active learning of Models 1 and 2',
                    epilog='...')
parser.add_argument('iteration', metavar='i', type=int, help='iterations number for the campaign')
args = parser.parse_args()
ITERATION = args.iteration # specify the current itereation number

DATA_DIR = "./output/"
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap/activephasemap/pretrained/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]
TOTAL_ITERATIONS  = len(glob.glob(DATA_DIR+"comp_model_*.json"))

# Load trained NP model for p(y|z)
np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
np_model.load_state_dict(torch.load(DATA_DIR+'np_model_%d.pt'%(ITERATION), map_location=device, weights_only=True))

# Load trained composition to latent model for p(z|c)
comp_model = XGBoost(xgb_model_args)
comp_model.load(DATA_DIR+"comp_model_%d.json"%ITERATION)

design_space_bounds = [(0.0, 35.0), (0.0, 35.0)]

# Create the experiment class to load all teh data
expt = UVVisExperiment(design_space_bounds, "./data/")
expt.read_iter_data(ITERATION)
expt.generate(use_spline=True)
bounds_np = expt.bounds.cpu().numpy()

""" 1. Create grid data """

def get_spectra(t, c, comp_model, np_model, nz=20):
    """Batch function to compute spectra.

    C shape : (n_samples, dimensions)
    """
    # predict z value from the comp model
    nr, dx = c.shape
    ct = torch.from_numpy(c).to(device).view(nr, 1, dx)
    z_mu, z_std = comp_model.predict(ct)   
    nr, nb, dz = z_mu.shape

    # sample z values from the dist
    z_dist = torch.distributions.Normal(z_mu, z_std)
    z = z_dist.rsample(torch.Size([nz])).view(nz*nr*nb, dz)

    time = torch.from_numpy(t).repeat(nz*nr*nb, 1, 1).to(device)
    time = torch.swapaxes(time, 1, 2)

    # sample spectra from NP model conditioned on z
    y_samples, _ = np_model.xz_to_y(time, z)

    return y_samples.view(nz, nr, nb, len(t)) 

@torch.no_grad()
def sample_grid(n_grid_spacing):
    grid_comps = get_twod_grid(n_grid_spacing, bounds_np)
    grid_spectra_samples = get_spectra(expt.t, grid_comps, comp_model, np_model)
    grid_spectra = np.zeros((grid_comps.shape[0], len(expt.t), 2))
    grid_spectra[...,0] = grid_spectra_samples.mean(dim=0).squeeze().cpu().numpy()
    grid_spectra[...,0] = grid_spectra_samples.std(dim=0).squeeze().cpu().numpy()

    return grid_comps, grid_spectra

grid_comps, grid_spectra = sample_grid(10)
np.savez("./paper/grid_data_10_%d.npz"%ITERATION, comps=grid_comps, spectra=grid_spectra)

if ITERATION==TOTAL_ITERATIONS:
    grid_comps, grid_spectra = sample_grid(20)
    np.savez("./paper/grid_data_20.npz", comps=grid_comps, spectra=grid_spectra)

    grid_comps, grid_spectra = sample_grid(30)
    np.savez("./paper/grid_data_30.npz", comps=grid_comps, spectra=grid_spectra)

""" 2. Create acqusition function data """

acqf = XGBUncertainity(expt, expt.bounds, np_model, comp_model)
C_grid = get_twod_grid(30, bounds_np)
with torch.no_grad():
    acq_values = acqf(torch.tensor(C_grid).reshape(len(C_grid),1,2).to(device)).squeeze().cpu().numpy()

np.savez("./paper/acqf_data_%d.npz"%ITERATION, comps=C_grid, values=acq_values)

""" 3. Create data for train and test errors """

def load_models_from_iteration(i):
    expt = UVVisExperiment(design_space_bounds, "./data/")
    expt.read_iter_data(i)
    expt.generate(use_spline=True)

    # Load trained NP model for p(y|z)
    np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
    np_model.load_state_dict(torch.load(DATA_DIR+'np_model_%d.pt'%(i), map_location=device, weights_only=True))

    # Load trained composition to latent model for p(z|c)
    comp_model = XGBoost(xgb_model_args)
    comp_model.load(DATA_DIR+"comp_model_%d.json"%i)

    return expt, comp_model, np_model

def minmax_normalize(tensor, dim=None):
    """
    Min-max normalize a tensor along a specified dimension or globally.
    
    :param tensor: Input tensor of shape (nz, nr, nb, not).
    :param dim: Dimension to normalize along. If None, normalize globally across all elements.
    :return: Min-max normalized tensor with the same shape.
    """
    if dim is None:
        # Global normalization across all elements
        min_val = tensor.min()
        max_val = tensor.max()
    else:
        # Normalize along the specified dimension
        min_val, _ = tensor.min(dim=dim, keepdim=True)
        max_val, _ = tensor.max(dim=dim, keepdim=True)
    
    # Prevent division by zero in case of constant values
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    
    return (tensor - min_val) / range_val

@torch.no_grad()
def get_accuracy(t, comps, spectra, comp_model, np_model):
    y_samples = get_spectra(t, comps, comp_model, np_model)
    y_samples_normed = minmax_normalize(y_samples, dim=-1)
    mu = y_samples_normed.mean(dim=0).squeeze()
    target = minmax_normalize(torch.from_numpy(spectra).to(device), dim=-1)
    
    loss = torch.nn.functional.mse_loss(target, mu, reduction='none').mean(dim=1)
    
    return loss.cpu().squeeze().numpy()


def get_accuraciy_plot_data():
    accuracies = {}
    for i in range(1,TOTAL_ITERATIONS+1):
        expt, comp_model, np_model = load_models_from_iteration(i)
        train_accuracy = get_accuracy(expt.t, 
                                      expt.comps.astype(np.double), 
                                      expt.spectra_normalized, 
                                      comp_model, 
                                      np_model
                                      )
        if not i==TOTAL_ITERATIONS:
            next_comps = np.load("./data/comps_%d.npy"%(i)).astype(np.double)
            next_spectra = np.load("./data/spectra_%d.npy"%(i))
            wav = np.load("./data/wav.npy")
            next_time = (wav-min(wav))/(max(wav)-min(wav))
            test_accuracy =  get_accuracy(next_time,
                                          next_comps, 
                                          next_spectra, 
                                          comp_model, 
                                          np_model
                                          )
            print("Iteration %d : Train error : %2.4f \t Test error : %2.4f"%(i, train_accuracy.mean(), test_accuracy.mean()))
        else:
            test_accuracy = np.nan
            print("Iteration %d : Train error : %2.4f"%(i, train_accuracy.mean()))

        accuracies[i] = {"train": train_accuracy, "test": test_accuracy}
        

    with open("./paper/accuracies.pkl", 'wb') as handle:
        pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 

if ITERATION==TOTAL_ITERATIONS:
    get_accuraciy_plot_data()
else:
    print("Total number of iterations %d is higher than current iteration run %d"%(TOTAL_ITERATIONS, ITERATION))