import os, sys, time, shutil, pdb, argparse, json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)

from activephasemap.models.np import NeuralProcess, context_target_split
from activephasemap.models.utils import finetune_neural_process
from activephasemap.models.mlp import MLP
from activephasemap.utils.acquisition import CompositeModelUncertainity
from activephasemap.utils.simulators import GNPPhases, UVVisExperiment
from activephasemap.utils.visuals import MinMaxScaler, _inset_spectra, scaled_tickformat, get_twod_grid
from activephasemap.utils.settings import initialize_points

RNG = np.random.default_rng()

import seaborn as sns 
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps 
from matplotlib.cm import ScalarMappable

parser = argparse.ArgumentParser(
                    prog='Train emulator of gold nanoparticle synthesis',
                    description='Perform a single iteration of active learning of Models 1 and 2',
                    epilog='...')
parser.add_argument('iteration', metavar='i', type=int, help='iterations number for the campaign')
args = parser.parse_args()
ITERATION = args.iteration # specify the current itereation number

# hyper-parameters
BATCH_SIZE = 12
N_INIT_POINTS = 24
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

mlp_model_args = {"num_epochs" : 100, 
                 "learning_rate" : 1e-2, 
                 "verbose": 25,
                 }

np_model_args = {"num_iterations": 500, 
                 "verbose":100, 
                 "lr":best_np_config["lr"], 
                 "batch_size": best_np_config["batch_size"]
                 }

@torch.no_grad
def print_matrix(A):
    A = pd.DataFrame(A.cpu().numpy())
    A.columns = ['']*A.shape[1]
    print(A.to_string())

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
  
    train_x = torch.from_numpy(comps_all)

    print("Std of latent variable ...: ")
    print_matrix(z_std)

    return train_x, z_mean, z_std

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

    train_x, train_z_mean, train_z_std = featurize_spectra(np_model, comps_all, spectra_all)
    comp_model = MLP(train_x, train_z_mean, train_z_std, **mlp_model_args)
    comp_loss = comp_model.fit()
    np.save(SAVE_DIR+'comp_loss_%d.npy'%ITERATION, comp_loss)
    torch.save(comp_model.state_dict(), SAVE_DIR+'comp_model_%d.pt'%ITERATION)

    print("Collecting next data points to sample by acqusition optimization...")
    acqf = CompositeModelUncertainity(expt.t, bounds, np_model, comp_model)
    new_x = acqf.optimize(BATCH_SIZE)

    torch.save(train_x.cpu(), SAVE_DIR+"train_x_%d.pt" %ITERATION)
    torch.save(train_z_mean.cpu(), SAVE_DIR+"train_z_mean_%d.pt" %ITERATION)
    torch.save(train_z_std.cpu(), SAVE_DIR+"train_z_std_%d.pt" %ITERATION)

    return new_x.cpu().numpy(), np_loss, np_model, comp_loss, comp_model, acqf, train_x

def generate_spectra(sim, comps):
    "This functions mimics the UV-Vis characterization module run"
    print("Generating spectra for iteration %d"%ITERATION, '\n')
    spectra = np.zeros((len(comps), sim.n_domain))
    for j, cj in enumerate(comps):
        spectra[j,:] = sim.simulate(cj)

    df = pd.DataFrame(spectra)

    return df

def from_comp_to_spectrum(t, c, comp_model, np_model):
    ci = torch.tensor(c).to(device)
    z_mu, z_std = comp_model.mlp(ci)
    z_dist = torch.distributions.Normal(z_mu, z_std)
    z = z_dist.sample(torch.Size([100]))
    t = torch.from_numpy(expt.t).repeat(100, 1, 1).to(device)
    t = torch.swapaxes(t, 1, 2)
    y_samples, _ = np_model.xz_to_y(t, z)

    mean_pred = y_samples.mean(dim=0, keepdim=True)
    sigma_pred = y_samples.std(dim=0, keepdim=True)
    mu_ = mean_pred.cpu().squeeze()
    sigma_ = sigma_pred.cpu().squeeze() 

    return mu_, sigma_   

def plot_model_accuracy(expt, comp_model, np_model):
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
        with torch.no_grad():
            mu, sigma = from_comp_to_spectrum(expt.t, expt.comps[i,:], comp_model, np_model)
            ax.plot(expt.wl, mu)
            ax.fill_between(expt.wl, mu-sigma, mu+sigma, color='grey')
        ax.scatter(expt.wl, expt.F[i], color='k')
        plt.savefig(iter_plot_dir+'%d.png'%(i))
        plt.close()

def plot_iteration(query_idx, expt, new_x, comp_model, np_model, acquisition, z_dim):
    layout = [['A1','A2', 'C', 'C'], 
              ['B1', 'B2', 'C', 'C']
              ]
    
    # plot selected points
    C_train = expt.points
    bounds =  expt.bounds.cpu().numpy()
    C_grid = get_twod_grid(20, bounds)
    fig, axs = plt.subplot_mosaic(layout, figsize=(4*4, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    axs['A1'].scatter(expt.comps[:,0], expt.comps[:,1], marker='x', color='k')
    axs['A1'].set_xlabel('C1', fontsize=20)
    axs['A1'].set_ylabel('C2', fontsize=20)    
    axs['A1'].set_title('C sampling')
    axs['A1'].set_xlim([bounds[0,0], bounds[1,0]])
    axs['A1'].set_ylim([bounds[0,1], bounds[1,1]])

    # plot acqf
    with torch.no_grad():
        C_grid_ = torch.tensor(C_grid).to(device).reshape(len(C_grid),1,2)
        acq_values = acquisition(C_grid_).squeeze().cpu().numpy()
    cmap = colormaps["magma"]
    norm = Normalize(vmin=min(acq_values), vmax = max(acq_values))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    axs['A2'].tricontourf(C_grid[:,0], C_grid[:,1], acq_values, cmap=cmap, norm=norm)
    axs['A2'].scatter(new_x[:,0], new_x[:,1], marker='x', color='k')    
    divider = make_axes_locatable(axs["A2"])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel('Acqusition value')
    axs['A2'].set_xlabel('C1', fontsize=20)
    axs['A2'].set_ylabel('C2', fontsize=20) 

    with torch.no_grad():
        for _ in range(5):
            ci = RNG.choice(C_train)
            mu, _ = from_comp_to_spectrum(expt.t, ci, comp_model, np_model)
            t_ = expt.t
            axs['B2'].plot(t_, mu.cpu().squeeze(), color='grey')
            axs['B2'].set_title('random sample p(y|c)')
            axs['B2'].set_xlabel('t', fontsize=20)
            axs['B2'].set_ylabel('f(t)', fontsize=20) 

            z_sample = torch.randn((1, z_dim)).to(device)
            t = torch.from_numpy(t_)
            t = t.view(1, t_.shape[0], 1).to(device)
            mu, _ = np_model.xz_to_y(t, z_sample)
            axs['B1'].plot(t_, mu.cpu().squeeze(), color='grey')
            axs['B1'].set_title('random sample p(y|z)')
            axs['B1'].set_xlabel('t', fontsize=20)
            axs['B1'].set_ylabel('f(t)', fontsize=20) 

    # plot the full evaluation on composition space
    ax = axs['C']
    bounds = expt.bounds.cpu().numpy()
    scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
    scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
    
    ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
    ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    C_grid = get_twod_grid(10, bounds)
    with torch.no_grad():
        for ci in C_grid:
            mu, sigma = from_comp_to_spectrum(expt.t, ci, comp_model, np_model)
            mu_ = mu.cpu().squeeze().numpy()
            sigma_ = sigma.cpu().squeeze().numpy()
            norm_ci = np.array([scaler_x.transform(ci[0]), scaler_y.transform(ci[1])])
            _inset_spectra(norm_ci, expt.t, mu_, sigma_, ax, show_sigma=True)
    ax.set_xlabel('C1', fontsize=20)
    ax.set_ylabel('C2', fontsize=20)

    return fig, axs

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
    comps_new, np_loss, np_model, gp_loss, comp_model, acquisition, train_x = run_iteration(expt)
    np.save(EXPT_DATA_DIR+'comps_%d.npy'%(ITERATION), comps_new)
    spectra = generate_spectra(sim, comps_new)
    np.save(EXPT_DATA_DIR+'spectra_%d.npy'%ITERATION, spectra)

    plot_model_accuracy(expt, comp_model, np_model)

    plot_iteration(ITERATION, expt, comps_new, comp_model, np_model, acquisition, N_LATENT)
    plt.savefig(PLOT_DIR+'itr_%d.png'%ITERATION)
    plt.close()
