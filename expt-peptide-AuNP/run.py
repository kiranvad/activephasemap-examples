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
from activephasemap.utils.simulators import UVVisExperiment
from activephasemap.utils.settings import *
from activephasemap.utils.visuals import *

ITERATION = 3 # specify the current itereation number

# hyper-parameters
BATCH_SIZE = 12
MODEL_NAME = "gp"
PRETRAIN_LOC = "../pretrained/uvvis.pt"
N_LATENT = 2
EXPT_DATA_DIR = "./data/"
SAVE_DIR = "./output/"
PLOT_DIR = "./plots/"
DESIGN_SPACE_DIM = 2

if ITERATION==0:
    for direc in [EXPT_DATA_DIR, SAVE_DIR, PLOT_DIR]:
        if os.path.exists(direc):
            shutil.rmtree(direc)
        os.makedirs(direc)

""" Set up design space bounds """
design_space_bounds = [(0.0, 87.0), (0.0,11.0)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

""" Create a GP model class for surrogate """
model_args = {"model":"gp",
"num_epochs" : 2000,
"learning_rate" : 1e-3
}

""" Helper functions """
def featurize_spectra(np_model, comps_all, spectra_all):
    """ Obtain latent space embedding from spectra.
    """
    num_draws = 25
    num_samples, n_domain = spectra_all.shape
    spectra = torch.zeros((num_samples, n_domain)).to(device)
    for i, si in enumerate(spectra_all):
        spectra[i] = torch.tensor(si).to(device)
    t = torch.linspace(0, 1, n_domain)
    t = t.repeat(num_samples, 1).to(device)
    train_x, train_y = [], []
    for _ in range(num_draws):
        with torch.no_grad():
            train_x.append(torch.from_numpy(comps_all).to(device))
            x_context, y_context, _, _ = context_target_split(t.unsqueeze(2), spectra.unsqueeze(2), 25, 25)
            z, _ = np_model.xy_to_mu_sigma(x_context, y_context) 
            train_y.append(z)

    return torch.cat(train_x), torch.cat(train_y) 

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

    standard_bounds = torch.tensor([(float(1e-5), 1.0) for _ in range(DESIGN_SPACE_DIM)]).transpose(-1, -2).to(device)
    gp_model = initialize_model(MODEL_NAME, model_args, DESIGN_SPACE_DIM, N_LATENT, device) 

    train_x, train_y = featurize_spectra(np_model, comps_all, spectra_all)
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


# Set up a synthetic data emulating an experiment
if ITERATION == 0:
    init_x = initialize_points(bounds, BATCH_SIZE, output_dim, device)
    comps_init = init_x.detach().cpu().numpy()
    np.save(EXPT_DATA_DIR+'comps_0.npy', comps_init)
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

    fig, axs = plot_iteration(ITERATION, test_function, test_function.sim.comps, gp_model, np_model, acquisition, N_LATENT)
    axs['A2'].scatter(comps_new[:,0], comps_new[:,1], marker='x', color='k')
    plt.savefig(PLOT_DIR+'itr_%d.png'%ITERATION)
    plt.close()

    plot_model_accuracy(PLOT_DIR, gp_model, np_model, test_function)

    fig, ax = plt.subplots(figsize=(4,4))
    plot_gpmodel_grid(ax, test_function, gp_model, np_model,
    num_grid_spacing=10, color='k', show_sigma=True, scale_axis=True)
    plt.savefig(PLOT_DIR+'predicted_phasemap_%d.png'%ITERATION)
    plt.close()