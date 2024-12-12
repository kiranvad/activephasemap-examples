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
def sample_grid(n_grid_spacing):
    grid_comps = get_twod_grid(n_grid_spacing, bounds_np)
    grid_spectra = np.zeros((grid_comps.shape[0], len(expt.t), 2))
    with torch.no_grad():
        for i, ci in enumerate(grid_comps):
            mu, sigma = from_comp_to_spectrum(expt.t, ci, comp_model, np_model)
            mu_ = mu.cpu().squeeze().numpy()
            sigma_ = sigma.cpu().squeeze().numpy()
            grid_spectra[i, :, 0] = mu_ 
            grid_spectra[i, :, 1] = sigma_

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

@torch.no_grad()
def get_accuracy(comps, domain, spectra, comp_model, np_model):
    mu, sigma = [], []
    for i in range(comps.shape[0]):
        mu_i,sigma_i = from_comp_to_spectrum(domain, comps[i,:], comp_model, np_model)
        mu.append(mu_i)
        sigma.append(sigma_i)
    
    mu = torch.stack(mu)
    sigma = torch.stack(sigma)
    target = torch.from_numpy(spectra)
    loss = torch.abs((target-mu)/(sigma+1e-8)).mean(dim=1)
    
    return loss.detach().cpu().squeeze().numpy()

def get_accuraciy_plot_data():
    accuracies = {}
    for i in range(1,TOTAL_ITERATIONS+1):
        expt, comp_model, np_model = load_models_from_iteration(i)
        train_accuracy = get_accuracy(expt.comps.astype(np.double), 
                                        expt.t, 
                                        expt.spectra_normalized, 
                                        comp_model, 
                                        np_model
                                        )
        if not i==TOTAL_ITERATIONS:
            next_comps = np.load("./data/comps_%d.npy"%(i)).astype(np.double)
            next_spectra = np.load("./data/spectra_%d.npy"%(i))
            wav = np.load("./data/wav.npy")
            next_time = (wav-min(wav))/(max(wav)-min(wav))

            test_accuracy =  get_accuracy(next_comps, 
                                          next_time, 
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