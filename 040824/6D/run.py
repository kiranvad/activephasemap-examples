import os, shutil, argparse, sys
import numpy as np
import matplotlib.pyplot as plt

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from activephasemap.models.utils import update_np
from activephasemap.np.neural_process import NeuralProcess 
from activephasemap.np.utils import context_target_split
from activephasemap.test_functions.phasemaps import ExperimentalTestFunction
from activephasemap.utils.simulators import UVVisExperiment
from activephasemap.utils.settings import *
from activephasemap.utils.visuals import *


parser = argparse.ArgumentParser(
                    prog='peptide mediated gold nanoparticle synthesis experiment',
                    description='Perform a single iteration of active learning of Models 1 and 2',
                    epilog='...')
parser.add_argument('iteration', metavar='i', type=int, help='iterations number for the campaign')
args = parser.parse_args()
ITERATION = args.iteration # specify the current itereation number
print("Running iteration %d"%ITERATION)
# hyper-parameters
BATCH_SIZE = 96
N_INIT_POINTS = 72
PRETRAIN_LOC = "../uvvis_np.pt"
N_LATENT = 2
DESIGN_SPACE_DIM = 5
EXPT_DATA_DIR = "./data/"
SAVE_DIR = "./output/"
PLOT_DIR = "./plots/"

if ITERATION==0:
    for direc in [EXPT_DATA_DIR, SAVE_DIR, PLOT_DIR]:
        if os.path.exists(direc):
            shutil.rmtree(direc)
        os.makedirs(direc)

""" Set up design space bounds """
design_space_bounds = [(0.0, 75.0),
                       (0.0, 75.0),
                       (0.0, 75.0),
                       (0.0, 75.0), 
                       (0.0, 11.0),
                       ]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

# you'd have to adjust the learning, number of iterations for early stopping 
# Optimizing GP hyper-parameters is a highly non-trivial case so you need to chose the 
# optimization algorithm parameters carefully.
gp_model_args = {"model":"gp", "num_epochs" : 1000, "learning_rate" : 1e-3, "verbose": 1}
np_model_args = {"num_iterations": 1000, "verbose":True, "print_freq":100, "lr":1e-3}

""" Helper functions """
def featurize_spectra(np_model, comps_all, spectra_all):
    """ Obtain latent space embedding from spectra.
    """
    num_draws = 20
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
    np_model = NeuralProcess(1, 1, 128, N_LATENT, 128).to(device)
    np_model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device))
    
    # Specify GP Model
    gp_model = initialize_model(gp_model_args, DESIGN_SPACE_DIM, N_LATENT, device) 

    # train np model to update its latent representation based on the new data
    np_model, np_loss = update_np(test_function.sim.t, spectra_all, np_model, **np_model_args)
    torch.save(np_model.state_dict(), SAVE_DIR+'np_model_%d.pt'%ITERATION)
    np.save(SAVE_DIR+'np_loss_%d.npy'%ITERATION, np_loss)

    # featurize spectra based on the updated NP Model
    train_x, train_y = featurize_spectra(np_model, comps_all, spectra_all)

    # train GP model given updated latent rep of np model
    normalized_x = normalize(train_x, bounds)
    gp_loss = gp_model.fit(normalized_x, train_y)
    np.save(SAVE_DIR+'gp_loss_%d.npy'%ITERATION, gp_loss)
    torch.save(gp_model.state_dict(), SAVE_DIR+'gp_model_%d.pt'%ITERATION)

    acquisition = construct_acqf_by_model(gp_model, normalized_x, train_y, N_LATENT)
    standard_bounds = torch.tensor([(float(1e-5), 1.0) for _ in range(DESIGN_SPACE_DIM)]).transpose(-1, -2).to(device)
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

    return new_x.cpu().numpy(), np_loss, np_model, gp_loss, gp_model, acquisition, train_x

# Set up a synthetic data emulating an experiment
if ITERATION == 0:
    init_x = initialize_points(bounds, N_INIT_POINTS, device)
    comps_init = init_x.detach().cpu().numpy()
    print("Compositions selected at itereation %d\n"%ITERATION, comps_init)
    np.save(EXPT_DATA_DIR+'comps_0.npy', comps_init)
else: 
    expt = UVVisExperiment(ITERATION, EXPT_DATA_DIR)
    expt.generate(use_spline=True)
    test_function = ExperimentalTestFunction(sim=expt, bounds=design_space_bounds)
    
    # plot train spectra data
    fig, ax = plt.subplots()
    for si in test_function.sim.spectra_normalized:
        ax.plot(test_function.sim.t, si)
    plt.savefig(PLOT_DIR+'train_spectra_%d.png'%ITERATION)
    plt.close()

    # obtain new set of compositions to synthesize and their spectra
    comps_new, np_loss, np_model, gp_loss, gp_model, acquisition, train_x = run_iteration(test_function)
    np.save(EXPT_DATA_DIR+'comps_%d.npy'%(ITERATION), comps_new)

    fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
    axs[0].plot(np.arange(len(gp_loss)), gp_loss)
    axs[1].plot(np.arange(len(np_loss)), np_loss)  
    plt.tight_layout()
    plt.savefig(PLOT_DIR+'itr_%d.png'%ITERATION)
    plt.close()      

    plot_model_accuracy(PLOT_DIR, gp_model, np_model, test_function)