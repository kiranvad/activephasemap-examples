import os, sys, time, shutil, pdb
from datetime import datetime
import numpy as np

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.penalized import PenalizedAcquisitionFunction

from activephasemap.models.hybrid import update_npmodel
from activephasemap.np.neural_process import NeuralProcess 
from activephasemap.test_functions.phasemaps import SimulatorTestFunction
from activephasemap.acquisitions.phaseboundary import PhaseBoundaryPenalty
from activephasemap.utils.simulators import PeptideGNPPhases
from activephasemap.utils.settings import *
from activephasemap.utils.visuals import *

BATCH_SIZE = 4
N_INIT_POINTS = 5
N_ITERATIONS = 10
TEMPERATURE = 35
MODEL_NAME = "dkl"
SIMULATOR = "goldnano"
DATA_DIR = "./peptide_grid/%d"%TEMPERATURE
PRETRAIN_LOC = "../pretrained/uvvis.pt"

SAVE_DIR = './%s_%d/'%(MODEL_NAME, TEMPERATURE)
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

sim = PeptideGNPPhases(DATA_DIR)
sim.generate()

N_LATENT = 2
design_space_bounds = [(0.0, 7.38), (0.0,7.27)]

np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
np_model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device))

test_function = SimulatorTestFunction(sim=sim, bounds=design_space_bounds)
input_dim = test_function.dim

init_x = initialize_points(test_function.bounds, N_INIT_POINTS, N_LATENT, device)
init_y, spectra = test_function(np_model, init_x)
bounds = test_function.bounds.to(device)

_bounds = [(0.0, 1.0) for _ in range(input_dim)]
standard_bounds = torch.tensor(_bounds).transpose(-1, -2).to(device)

train_x = init_x
train_y = init_y

if MODEL_NAME=="gp":
    model_args = {"model":"gp",
    "num_epochs" : 2000,
    "learning_rate" : 1e-3
    }
elif MODEL_NAME=="dkl":
    model_args = {"model": "dkl",
    "regnet_dims": [16,16,16],
    "regnet_activation": "tanh",
    "pretrain_steps": 0,
    "train_steps": 1000
    }

t = time.time()
data = ActiveLearningDataset(train_x,spectra)
for i in range(N_ITERATIONS):
    print("\niteration %d" % i)
    gp_model = initialize_model(MODEL_NAME, model_args, input_dim, N_LATENT, device)

    # fit model on normalized x
    normalized_x = normalize(train_x, bounds).to(train_x)
    gp_model.fit_and_save(normalized_x, train_y) 

    acqf_model = construct_acqf_by_model(gp_model, normalized_x, train_y, N_LATENT)
    acqf_phase = PhaseBoundaryPenalty(test_function, gp_model, np_model)

    # # Create an acqusition function with uncertainties from prediction and boundaries
    # acquisition = PenalizedAcquisitionFunction(
    #     acqf_model,
    #     penalty_func=acqf_phase,
    #     regularization_parameter=-1.0
    #     )

    acquisition = acqf_model

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

    new_x = unnormalize(normalized_candidates.detach(), bounds=bounds)
    # evaluate new y values and save
    new_y, new_spectra = test_function(np_model, new_x)

    if np.remainder(100*(i+1)/N_ITERATIONS,10)==0:
        torch.save(gp_model.state_dict(), SAVE_DIR+'gp_model_%d.pt'%i)
        plot_experiment(test_function.sim.t, design_space_bounds, data)
        plt.savefig(SAVE_DIR+'train_spectra_%d.png'%i)
        # update np model with new data
        np_model, np_loss = update_npmodel(test_function.sim.t, np_model, data, num_iterations=75, verbose=False)
        torch.save(np_model.state_dict(), SAVE_DIR+'np_model_%d.pt'%i)
        plot_iteration(i, test_function, train_x, gp_model, np_model, acquisition, N_LATENT)
        plt.savefig(SAVE_DIR+'itr_%d.png'%i)
        plt.close()
        plot_gpmodel(test_function, gp_model, np_model, SAVE_DIR+'gpmodel_itr_%d.png'%i)
        plot_phasemap_pred(test_function, gp_model, np_model, SAVE_DIR+'compare_spectra_pred_%d.png'%i)
        plot_autophasemap(acqf_phase, SAVE_DIR+'autphasemap_%d.png'%i)
        torch.save(train_x.cpu(), SAVE_DIR+"train_x_%d.pt" %i)
        torch.save(train_y.cpu(), SAVE_DIR+"train_y_%d.pt" %i)

    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    data.update(new_x, new_spectra)

"""Plotting after training"""
gp_model = initialize_model(MODEL_NAME, model_args, input_dim, N_LATENT, device)
normalized_x = normalize(train_x, bounds).to(train_x)
gp_model.fit_and_save(normalized_x, train_y) 
torch.save(gp_model.state_dict(), SAVE_DIR+'gp_model.pt')
np_model, np_loss = update_npmodel(test_function.sim.t, np_model, data, num_iterations=75, verbose=False)
torch.save(np_model.state_dict(), SAVE_DIR+'np_model.pt')

torch.save(train_x.cpu(), "%s/train_x.pt" % SAVE_DIR)
torch.save(train_y.cpu(), "%s/train_y.pt" % SAVE_DIR)
