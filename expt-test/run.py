import os, sys, time, shutil, pdb
from datetime import datetime
import numpy as np
import pandas as pd

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from activephasemap.models.hybrid import update_npmodel
from activephasemap.np.neural_process import NeuralProcess 
from activephasemap.test_functions.phasemaps import ExperimentalTestFunction
from activephasemap.acquisitions.phaseboundary import PhaseBoundaryPenalty
from activephasemap.utils.simulators import GNPPhases, UVVisExperiment
from activephasemap.utils.settings import *
from activephasemap.utils.visuals import *

ITERATION = 4 # specify the current itereation number

# hyper-parameters
BATCH_SIZE = 4
N_INIT_POINTS = 5
N_ITERATIONS = 10
MODEL_NAME = "gp"
SIMULATOR = "goldnano"
DATA_DIR = "../AuNP/gold_nano_grid/"
PRETRAIN_LOC = "../pretrained/uvvis.pt"
N_LATENT = 2

EXPT_DATA_DIR = "./data/"
SAVE_DIR = './output/'
PLOT_DIR = './plots/'

if ITERATION==0:
    for direc in [EXPT_DATA_DIR, SAVE_DIR, PLOT_DIR]:
        shutil.rmtree(direc)
        os.makedirs(direc)

# setup a simulator to use as a proxy for experiment
sim = GNPPhases(DATA_DIR)
sim.generate()

""" Set up design space bounds """
input_dim = 2 # dimension of design space
output_dim = N_LATENT
design_space_bounds = [(0.0, 7.38), (0.0,7.27)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

""" Create a GP model class for surrogate """
model_args = {"model":"gp",
"num_epochs" : 2000,
"learning_rate" : 1e-3
}

""" Helper functions """
def featurize_spectra(np_model, spectra_all):
    """ Obtain latent space embedding from spectra.
    """
    num_samples, n_domain = spectra_all.shape
    spectra = torch.zeros((num_samples, n_domain)).to(device)
    for i, si in enumerate(spectra_all):
        spectra[i] = torch.tensor(si).to(device)
    t = torch.linspace(0, 1, n_domain)
    t = t.repeat(num_samples, 1).to(device)
    with torch.no_grad():
        z, _ = np_model.xy_to_mu_sigma(t.unsqueeze(2), spectra.unsqueeze(2)) 

    return z  

def run_iteration(test_function):
    """ Perform a single iteration of active phasemapping.

    helper function to run a single iteration given 
    all the compositions and spectra obtained so far. 
    """
    # assemble data for surrogate model training  
    comps_all = test_function.sim.comps 
    spectra_all = test_function.sim.spectra_normalized # use normalized spectra
    print('Data shapes : ', comps_all.shape, spectra_all.shape)

    # Specify the Neural Process model
    np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
    np_model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device))

    standard_bounds = torch.tensor([(float(1e-5), 1.0) for _ in range(input_dim)]).transpose(-1, -2).to(device)
    gp_model = initialize_model(MODEL_NAME, model_args, input_dim, output_dim, device) 

    train_x = torch.from_numpy(comps_all).to(device) 
    train_y = featurize_spectra(np_model, spectra_all)
    normalized_x = normalize(train_x, bounds)
    gp_model.fit_and_save(normalized_x, train_y)
    torch.save(gp_model.state_dict(), SAVE_DIR+'gp_model_%d.pt'%ITERATION)

    acquisition = construct_acqf_by_model(gp_model, normalized_x, train_y, N_LATENT)

    normalized_candidates, acqf_values = optimize_acqf(
        acquisition, 
        standard_bounds, 
        q=BATCH_SIZE, 
        num_restarts=20, 
        raw_samples=1024, 
        return_best_only=True,
        sequential=False,
        options={"batch_limit": 1, "maxiter": 10, "with_grad":True}
        )

    # calculate acquisition values after rounding
    new_x = unnormalize(normalized_candidates.detach(), bounds=bounds) 

    torch.save(train_x.cpu(), SAVE_DIR+"train_x_%d.pt" %ITERATION)
    torch.save(train_y.cpu(), SAVE_DIR+"train_y_%d.pt" %ITERATION)
    data = ActiveLearningDataset(train_x,spectra_all)
    np_model, np_loss = update_npmodel(test_function.sim.t, np_model, data, num_iterations=75, verbose=False)
    torch.save(np_model.state_dict(), SAVE_DIR+'np_model_%d.pt'%ITERATION)

    return new_x.cpu().numpy(), np_model, gp_model, acquisition, train_x

def generate_spectra(sim, comps):
    "This functions mimics the UV-Vis characterization module run"
    print("Generating spectra for iteration %d"%ITERATION, '\n', comps)
    spectra = np.zeros((len(comps), sim.n_domain))
    for j, cj in enumerate(comps):
        spectra[j,:] = sim.simulate(cj)

    df = pd.DataFrame(spectra)

    return df

# Set up a synthetic data emulating an experiment
if ITERATION == 0:
    init_x = initialize_points(bounds, N_INIT_POINTS, output_dim, device)
    comps_init = init_x.detach().cpu().numpy()
    np.save(EXPT_DATA_DIR+'comps_0.npy', comps_init)
    np.save(EXPT_DATA_DIR+'wav.npy', sim.wl_)
    spectra = generate_spectra(sim, comps_init)
    np.save(EXPT_DATA_DIR+'spectra_%d.npy'%ITERATION, spectra)
else: 
    expt = UVVisExperiment(ITERATION, EXPT_DATA_DIR)
    expt.generate()
    test_function = ExperimentalTestFunction(sim=expt, bounds=design_space_bounds)
    fig, ax = plt.subplots()
    expt.plot(ax, design_space_bounds)
    plt.savefig(PLOT_DIR+'train_spectra_%d.png'%ITERATION)
    plt.close()

    # obtain new set of compositions to synthesize and their spectra
    comps_new, np_model, gp_model, acquisition, train_x = run_iteration(test_function)
    np.save(EXPT_DATA_DIR+'comps_%d.npy'%(ITERATION), comps_new)
    spectra = generate_spectra(sim, comps_new)
    np.save(EXPT_DATA_DIR+'spectra_%d.npy'%ITERATION, spectra)

    plot_iteration(ITERATION, test_function, train_x, gp_model, np_model, acquisition, N_LATENT)
    plt.savefig(PLOT_DIR+'itr_%d.png'%ITERATION)
    plt.close()

    plot_model_accuracy(PLOT_DIR, gp_model, np_model, test_function)

    fig, ax = plt.subplots(figsize=(4,4))
    plot_gpmodel_grid(ax, test_function, gp_model, np_model,
    num_grid_spacing=10, color='k', show_sigma=True, scale_axis=True)
    plt.savefig(PLOT_DIR+'predicted_phasemap_%d.png'%ITERATION)
    plt.close()
