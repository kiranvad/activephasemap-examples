import os, sys, time, shutil, pdb, collections
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from activephasemap.models.anp import NeuralProcessModel
from activephasemap.models.utils import update_anp
from activephasemap.test_functions.phasemaps import ExperimentalTestFunction
from activephasemap.utils.simulators import GNPPhases, UVVisExperiment 
from activephasemap.utils.settings import initialize_points, initialize_model, construct_acqf_by_model

from helpers import *

ITERATION = 1 # specify the current itereation number

ITR = collections.namedtuple(
    "ITR",
    ("num","new_x", "np_model", "np_loss", "gp_model", "gp_loss", "acquisition", "train_x", "train_y"))

# hyper-parameters
hyper_params = {"batch_size" : 4,
"n_init_poitns" : 5,
"n_iterations" : 10,
"model_name" : "gp",
"simulator" : "goldnano",
"data_dir" : "../AuNP/gold_nano_grid/",
"pretrain_loc" : "../pretrained/ANP/uvvis_anp.pt",
"rep_dim" : 4,
"latent_dim" : 2,
"design_space_dim" : 2
} 

EXPT_DATA_DIR = "./data/"
SAVE_DIR = './output/'
PLOT_DIR = './plots/'

if ITERATION==0:
    for direc in [EXPT_DATA_DIR, SAVE_DIR, PLOT_DIR]:
        if os.path.exists(direc):
            shutil.rmtree(direc)
        os.makedirs(direc)

# setup a simulator to use as a proxy for experiment
sim = GNPPhases(hyper_params["data_dir"])
sim.generate()

""" Set up design space bounds """
design_space_bounds = [(0.0, 7.38), (0.0,7.27)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

""" Create a GP model class for surrogate """
model_args = {"model":"gp",
"num_epochs" : 1000,
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
            context_x, context_y, target_x, target_y = get_contex_target(t.unsqueeze(2), spectra.unsqueeze(2),50)
            r = np_model.r_encoder(context_x, context_y, target_x)
            q_target = np_model.z_encoder(target_x, target_y)
            z = q_target.rsample()
            rz = torch.cat([r.mean(dim=1),z], dim=-1)
            train_y.append(rz)

    return torch.cat(train_x), torch.cat(train_y) 

def run_iteration(test_function, hyper_params):
    """ Perform a single iteration of active phasemapping.

    helper function to run a single iteration given 
    all the compositions and spectra obtained so far. 
    """
    # assemble data for surrogate model training  
    comps_all = test_function.sim.comps 
    spectra_all = test_function.sim.spectra_normalized # use normalized spectra
    print('Data shapes : ', comps_all.shape, spectra_all.shape)

    # Specify the Neural Process model
    np_model = NeuralProcessModel(hyper_params["rep_dim"], hyper_params["latent_dim"], attn_type="multihead").to(device)
    np_model.load_state_dict(torch.load(hyper_params["pretrain_loc"], map_location=device))

    standard_bounds = torch.tensor([(float(1e-5), 1.0) for _ in range(hyper_params["design_space_dim"])]).transpose(-1, -2).to(device)
    n_latent = hyper_params["rep_dim"]+hyper_params["latent_dim"]
    gp_model = initialize_model(hyper_params["model_name"],model_args, hyper_params["design_space_dim"], n_latent, device) 

    train_x, train_y = featurize_spectra(np_model, comps_all, spectra_all)
    normalized_x = normalize(train_x, bounds)
    gp_loss = gp_model.fit_and_save(normalized_x, train_y)
    torch.save(gp_model.state_dict(), SAVE_DIR+'gp_model_%d.pt'%ITERATION)

    acquisition = construct_acqf_by_model(gp_model, normalized_x, train_y, n_latent)

    normalized_candidates, acqf_values = optimize_acqf(
        acquisition, 
        standard_bounds, 
        q=hyper_params["batch_size"], 
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
    np_model, np_loss = update_anp(test_function.sim.t, spectra_all, np_model)
    torch.save(np_model.state_dict(), SAVE_DIR+'np_model_%d.pt'%ITERATION)

    return ITR(num = ITERATION,
    new_x = new_x.cpu().numpy(), 
    np_model = np_model, 
    np_loss = np_loss,
    gp_model = gp_model, 
    gp_loss = gp_loss,
    acquisition = acquisition, 
    train_x = train_x,
    train_y = train_y,
    )

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
    init_x = initialize_points(bounds, N_INIT_POINTS, device)
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
    itr = run_iteration(test_function, hyper_params)
    comps_new = itr.new_x
    np.save(EXPT_DATA_DIR+'comps_%d.npy'%(ITERATION), comps_new)
    spectra = generate_spectra(sim, comps_new)
    np.save(EXPT_DATA_DIR+'spectra_%d.npy'%ITERATION, spectra)

    fig, axs = plot_iteration(itr, test_function, hyper_params)
    axs['A2'].scatter(comps_new[:,0], comps_new[:,1],color='k')
    plt.savefig(PLOT_DIR+'itr_%d.png'%ITERATION)
    plt.close()

    plot_model_accuracy(PLOT_DIR, itr, test_function)

    fig, ax = plt.subplots(figsize=(4,4))
    plot_gpmodel_grid(ax, test_function, itr,
    num_grid_spacing=10, color='k', show_sigma=True, scale_axis=True)
    plt.savefig(PLOT_DIR+'predicted_phasemap_%d.png'%ITERATION)
    plt.close()
