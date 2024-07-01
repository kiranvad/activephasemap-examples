import os, shutil, argparse, json, time, datetime, pdb
import numpy as np
import matplotlib.pyplot as plt

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from activephasemap.models.utils import finetune_neural_process
from activephasemap.models.np import NeuralProcess, context_target_split
from activephasemap.models.gp import MultiTaskGP
from activephasemap.utils.simulators import UVVisExperiment
from activephasemap.utils.settings import *
start = time.time()
parser = argparse.ArgumentParser(
                    prog='peptide mediated gold nanoparticle synthesis experiment',
                    description='Perform a single iteration of active learning of Models 1 and 2',
                    epilog='...')
parser.add_argument('iteration', metavar='i', type=int, help='iterations number for the campaign')
args = parser.parse_args()
ITERATION = args.iteration # specify the current itereation number
print("Running iteration %d"%ITERATION)
# hyper-parameters
BATCH_SIZE = 87
N_INIT_POINTS = 72
DESIGN_SPACE_DIM = 5

EXPT_DATA_DIR = "./data/"
SAVE_DIR = "./output/"
PLOT_DIR = "./plots/"

with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]

PRETRAIN_LOC = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/test_np_new_api/model.pt"

if ITERATION==0:
    for direc in [EXPT_DATA_DIR, SAVE_DIR, PLOT_DIR]:
        if os.path.exists(direc):
            shutil.rmtree(direc)
        os.makedirs(direc)
    os.makedirs(PLOT_DIR+'preds/')
""" Set up design space bounds """
design_space_bounds = [(0.0, 75.0),
                       (0.0, 75.0),
                       (0.0, 75.0),
                       (0.0, 75.0), 
                       (0.0, 11.0),
                       ]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

gp_model_args = {"model":"gp", 
                 "num_epochs" : 500, 
                 "learning_rate" : 5e-1, 
                 "verbose": 100
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
    num_draws = 32
    num_samples, n_domain = spectra_all.shape
    spectra = torch.zeros((num_samples, n_domain)).to(device)
    for i, si in enumerate(spectra_all):
        spectra[i] = torch.tensor(si).to(device)
    t = torch.linspace(0, 1, n_domain)
    t = t.repeat(num_samples, 1).to(device)
    train_y, train_y_std = [], []
    train_x = torch.from_numpy(comps_all).to(device)
    y = []
    for _ in range(num_draws):
        with torch.no_grad():
            x_context, y_context, _, _ = context_target_split(t.unsqueeze(2), spectra.unsqueeze(2), 25, 25)
            z, _ = np_model.xy_to_mu_sigma(x_context, y_context) 
        y.append(z)

    train_y = torch.stack(y).mean(dim=0).to(device)
    train_y_std = torch.stack(y).std(dim=0).to(device)

    return train_x, train_y, train_y_std

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

    train_x, train_y, train_y_std = featurize_spectra(np_model, comps_all, spectra_all)
    normalized_x = normalize(train_x, bounds).to(device)
    print("GP data shapes : ", normalized_x.shape, train_y.shape)
    gp_model = MultiTaskGP(normalized_x, train_y, gp_model_args, DESIGN_SPACE_DIM, N_LATENT, train_y_std) 
    gp_loss = gp_model.fit() 
    torch.save(gp_model.state_dict(), SAVE_DIR+'gp_model_%d.pt'%ITERATION)
    np.save(SAVE_DIR+'gp_loss_%d.npy'%ITERATION, gp_loss)

    print("Collecting next data points to sample by acqusition optimization...")
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


    return new_x.cpu().numpy(), np_loss, np_model, gp_loss, gp_model

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
    print("Compositions selected at itereation %d\n"%ITERATION, comps_init)
    np.save(EXPT_DATA_DIR+'comps_0.npy', comps_init)
else: 
    expt = UVVisExperiment(design_space_bounds, ITERATION, EXPT_DATA_DIR)
    expt.generate(use_spline=True)

    fig, ax = plt.subplots()
    expt.plot(ax, design_space_bounds)
    plt.savefig(PLOT_DIR+'train_spectra_%d.png'%ITERATION)
    plt.close()

    # obtain new set of compositions to synthesize and their spectra
    comps_new, np_loss, np_model, gp_loss, gp_model = run_iteration(expt)
    # np.save(EXPT_DATA_DIR+'comps_%d.npy'%(ITERATION), comps_new)

    fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
    axs[0].plot(np.arange(len(np_loss)), np_loss)
    axs[0].set_title("NP-Loss")  
    axs[1].plot(np.arange(len(gp_loss)), gp_loss)
    axs[1].set_title("GP-Loss")
    plt.savefig(PLOT_DIR+'loss_%d.png'%ITERATION)
    plt.close()      

    plot_model_accuracy(expt, gp_model, np_model)

end = time.time()
time_str =  str(datetime.timedelta(seconds=end-start)) 
print('Total time : %s'%(time_str))