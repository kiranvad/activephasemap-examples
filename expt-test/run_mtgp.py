import os, sys, time, shutil, pdb, argparse, json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)

from botorch.utils.transforms import normalize, unnormalize
from activephasemap.models.mtgp import MultiTaskGPVersion2 as GaussianProcess
from activephasemap.models.np import NeuralProcess, context_target_split
from activephasemap.utils.acquisition import UncertainitySelector
from activephasemap.models.utils import finetune_neural_process
from activephasemap.utils.simulators import GNPPhases, UVVisExperiment
from activephasemap.utils.settings import *
from activephasemap.utils.visuals import *

parser = argparse.ArgumentParser(
                    prog='Train emulator of gold nanoparticle synthesis',
                    description='Perform a single iteration of active learning of Models 1 and 2',
                    epilog='...')
parser.add_argument('iteration', metavar='i', type=int, help='iterations number for the campaign')
args = parser.parse_args()
ITERATION = args.iteration # specify the current itereation number

# hyper-parameters
BATCH_SIZE = 4
N_INIT_POINTS = 5
N_ITERATIONS = 10
MODEL_NAME = "gp"
SIMULATOR = "goldnano"
DATA_DIR = "../AuNP/gold_nano_grid/"
DESIGN_SPACE_DIM = 2
NUM_Z_DRAWS = 16
EXPT_DATA_DIR = "./data/"
SAVE_DIR = './output/'
PLOT_DIR = './plots/'

with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]

PRETRAIN_LOC = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/test_np_new_api/model.pt"

if ITERATION==0:
    for direc in [EXPT_DATA_DIR, SAVE_DIR, PLOT_DIR]:
        if os.path.exists(direc):
            shutil.rmtree(direc)
        os.makedirs(direc)

# setup a simulator to use as a proxy for experiment
sim = GNPPhases(DATA_DIR)
sim.generate()

""" Set up design space bounds """
design_space_bounds = [(0.0, 7.38), (0.0,7.27)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

gp_model_args = {"num_epochs" : 1000, 
                 "learning_rate" : 1e-2, 
                 "verbose": 50,
                 }

np_model_args = {"num_iterations": 500, 
                 "verbose":100, 
                 "lr":best_np_config["lr"], 
                 "batch_size": best_np_config["batch_size"]
                 }


""" Helper functions """
def featurize_spectra(np_model, comps_all, spectra_all):
    """ Obtain latent space embedding from spectra.
    """
    num_samples, n_domain = spectra_all.shape
    spectra = torch.zeros((num_samples, n_domain)).to(device)
    for i, si in enumerate(spectra_all):
        spectra[i] = torch.tensor(si).to(device)
    t = torch.linspace(0, 1, n_domain)
    t = t.repeat(num_samples, 1).to(device)
    with torch.no_grad():
        z_mean, z_std = np_model.xy_to_mu_sigma(t.unsqueeze(2), spectra.unsqueeze(2)) 
        z_dist = torch.distributions.normal.Normal(z_mean, z_std)
  
    train_x = torch.from_numpy(comps_all).repeat(NUM_Z_DRAWS, 1)
    train_y = z_dist.sample(torch.Size([NUM_Z_DRAWS]))
    train_y = train_y.reshape(num_samples*NUM_Z_DRAWS, z_dist.mean.shape[1])

    return train_x.to(device), train_y.to(device)

def run_iteration(expt):
    """ Perform a single iteration of active phasemapping.

    helper function to run a single iteration given 
    all the compositions and spectra obtained so far. 
    """
    # assemble data for surrogate model training  
    comps_all = expt.comps 
    spectra_all = expt.spectra_normalized # use normalized spectra
    print('Data shapes : ', comps_all.shape, spectra_all.shape)

    # Specify the Neural Process model
    np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
    np_model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device))

    np_model, np_loss = finetune_neural_process(expt.t, spectra_all, np_model, **np_model_args)
    torch.save(np_model.state_dict(), SAVE_DIR+'np_model_%d.pt'%ITERATION)
    np.save(SAVE_DIR+'np_loss_%d.npy'%ITERATION, np_loss)

    train_x, train_y = featurize_spectra(np_model, comps_all, spectra_all)
    normalized_x = normalize(train_x, bounds)
    gp_model = GaussianProcess(normalized_x, train_y, **gp_model_args)
    gp_loss = gp_model.fit()
    np.save(SAVE_DIR+'gp_loss_%d.npy'%ITERATION, gp_loss)
    torch.save(gp_model.state_dict(), SAVE_DIR+'gp_model_%d.pt'%ITERATION)

    print("Collecting next data points to sample by acqusition optimization...")
    acqf = UncertainitySelector(expt.dim, gp_model, bounds)
    new_x = acqf.optimize(BATCH_SIZE)

    torch.save(train_x.cpu(), SAVE_DIR+"train_x_%d.pt" %ITERATION)
    torch.save(train_y.cpu(), SAVE_DIR+"train_y_%d.pt" %ITERATION)

    return new_x.cpu().numpy(), np_loss, np_model, gp_loss, gp_model, acqf, train_x

def generate_spectra(sim, comps):
    "This functions mimics the UV-Vis characterization module run"
    print("Generating spectra for iteration %d"%ITERATION, '\n')
    spectra = np.zeros((len(comps), sim.n_domain))
    for j, cj in enumerate(comps):
        spectra[j,:] = sim.simulate(cj)

    df = pd.DataFrame(spectra)

    return df

def plot_model_accuracy(expt, gp_model, np_model):
    """ Plot accuracy of model predictions of experimental data

    This provides a qualitative understanding of current model 
    on training data.
    """
    print("Creating plots to visualize training data predictions...")
    iter_plot_dir = PLOT_DIR+'preds_%d/'%ITERATION
    if os.path.exists(iter_plot_dir):
        shutil.rmtree(iter_plot_dir)
    os.makedirs(iter_plot_dir)

    num_samples, c_dim = expt.comps.shape
    for i in range(num_samples):
        fig, ax = plt.subplots()
        ci = expt.comps[i,:].reshape(1, c_dim)
        with torch.no_grad():
            mu, sigma = from_comp_to_spectrum(expt, gp_model, np_model, ci)
            mu_ = mu.cpu().squeeze()
            sigma_ = sigma.cpu().squeeze()
            ax.plot(expt.wl, mu_)
            ax.fill_between(expt.wl, mu_-sigma_, mu_+sigma_, color='grey')
        ax.scatter(expt.wl, expt.F[i], color='k')
        plt.savefig(iter_plot_dir+'%d.png'%(i))
        plt.close()

# Set up a synthetic data emulating an experiment
if ITERATION == 0:
    init_x = initialize_points(bounds, N_INIT_POINTS, device)
    comps_init = init_x.detach().cpu().numpy()
    np.save(EXPT_DATA_DIR+'comps_0.npy', comps_init)
    np.save(EXPT_DATA_DIR+'wav.npy', sim.wl)
    spectra = generate_spectra(sim, comps_init)
    np.save(EXPT_DATA_DIR+'spectra_%d.npy'%ITERATION, spectra)
else: 
    expt = UVVisExperiment(design_space_bounds, ITERATION, EXPT_DATA_DIR)
    expt.generate(use_spline=True)

    fig, ax = plt.subplots()
    expt.plot(ax, design_space_bounds)
    plt.savefig(PLOT_DIR+'train_spectra_%d.png'%ITERATION)
    plt.close()

    # obtain new set of compositions to synthesize and their spectra
    comps_new, np_loss, np_model, gp_loss, gp_model, acquisition, train_x = run_iteration(expt)
    np.save(EXPT_DATA_DIR+'comps_%d.npy'%(ITERATION), comps_new)
    spectra = generate_spectra(sim, comps_new)
    np.save(EXPT_DATA_DIR+'spectra_%d.npy'%ITERATION, spectra)

    fig, axs = plt.subplots(1,2,figsize=(2*4, 4))
    axs[0].plot(np.arange(len(gp_loss)), gp_loss)
    axs[1].plot(np.arange(len(np_loss)), np_loss)  
    plt.savefig(PLOT_DIR+'loss_%d.png'%ITERATION)
    plt.close()      

    plot_iteration(ITERATION, expt, comps_new,  gp_model, np_model, acquisition, N_LATENT)
    plt.savefig(PLOT_DIR+'itr_%d.png'%ITERATION)
    plt.close()

    plot_model_accuracy(expt, gp_model, np_model)
    
    plot_gpmodel_expt(expt, gp_model, np_model)
    plt.tight_layout()
    plt.savefig(PLOT_DIR+'gp_model_%d.png'%ITERATION)
    plt.close()

    fig, ax = plt.subplots(figsize=(4,4))
    plot_gpmodel_grid(ax, expt, gp_model, np_model,
                      num_grid_spacing=10, color='k', show_sigma=True, scale_axis=True
                      )
    plt.tight_layout()
    plt.savefig(PLOT_DIR+'predicted_phasemap_%d.png'%ITERATION)
    plt.close()
