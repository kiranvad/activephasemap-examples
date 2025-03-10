import torch
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import pandas as pd 
import pdb, argparse, json, glob, pickle, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from activephasemap.models.np import NeuralProcess
from activephasemap.simulators import UVVisExperiment
from activephasemap.models.xgb import XGBoost
from activephasemap.utils import *
from apdist.distances import AmplitudePhaseDistance as dist
from apdist.geometry import SquareRootSlopeFramework as SRSF
from scipy.ndimage import gaussian_filter

DATA_DIR = "../output/"
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap/activephasemap/pretrained/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]
TOTAL_ITERATIONS  = len(glob.glob(DATA_DIR+"comp_model_*.json"))
design_space_bounds = [(0.0, 35.0), (0.0, 35.0)]

eval_comps = np.load("./comps_evals.npy")
eval_spectra = np.load("./spectra_evals.npy")
print("Evaluation data shapes : ", eval_comps.shape, eval_spectra.shape)
wav = np.load("../data/wav.npy")
wav_normalized = (wav-min(wav))/(max(wav)-min(wav))

def load_models_from_iteration(i):
    expt = UVVisExperiment(design_space_bounds, "../data/")
    expt.read_iter_data(i)
    expt.generate(use_spline=True)

    # Load trained NP model for p(y|z)
    np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
    np_model.load_state_dict(torch.load(DATA_DIR+'np_model_%d.pt'%(i), map_location=device, weights_only=True))

    # Load trained composition to latent model for p(z|c)
    comp_model = XGBoost(xgb_model_args)
    comp_model.load(DATA_DIR+"comp_model_%d.json"%i)

    return expt, comp_model, np_model

def smoothen_and_normalize(y):
    y_hat = gaussian_filter(y,  sigma=1.0)
    y_hat_norm =  (y_hat-min(y_hat))/(max(y_hat)-min(y_hat))

    return y_hat_norm

def weighted_amplitude_phase(x, y_ref, y_query, **kwargs):
    srsf = SRSF(x)
    q_ref = srsf.to_srsf(smoothen_and_normalize(y_ref))
    q_query = srsf.to_srsf(smoothen_and_normalize(y_query))
    gamma = srsf.get_gamma(q_ref, q_query, **kwargs)

    delta = q_ref-q_query
    if delta.sum() == 0:
        dist = 0
    else:
        gam_dev = np.gradient(gamma, srsf.time)
        q_gamma = np.interp(gamma, srsf.time, q_query)
        y_amplitude = (q_ref - (q_gamma * np.sqrt(gam_dev))) ** 2

        amplitude = np.sqrt(np.trapz(y_amplitude, srsf.time))

        p_gamma = np.sqrt(gam_dev)*y_ref # we define p(\gamma) = \sqrt{\dot{\gamma(t)}} * f(t)
        p_identity = np.ones_like(gam_dev)*y_ref
        y_phase =  (p_gamma - p_identity) ** 2

        phase = np.sqrt(np.trapz(y_phase, srsf.time))

    return amplitude, phase

@torch.no_grad()
def get_accuracy(comps, domain, spectra, comp_model, np_model):
    loss = []
    optim_kwargs = {"optim":"DP", "grid_dim":10}
    for i in range(comps.shape[0]):
        mu_i, _ = from_comp_to_spectrum(domain, comps[i,:], comp_model, np_model)
        mu_i_np = mu_i.cpu().squeeze().numpy()
        amplitude, phase = weighted_amplitude_phase(domain, spectra[i,:], mu_i_np)
        loss.append(0.5*(amplitude+phase))

    return np.asarray(loss)

@torch.no_grad()
def plot_eval_comparision(domain, comps, spectra, comp_model, np_model, direc):
    eval_plot_dir = direc+'eval_compare/'
    if os.path.exists(eval_plot_dir):
        shutil.rmtree(eval_plot_dir)
    os.makedirs(eval_plot_dir)
    for i in range(comps.shape[0]):
        fig, ax = plt.subplots()
        mu, sigma = from_comp_to_spectrum(domain, comps[i,:], comp_model, np_model)
        mu_np = mu.cpu().squeeze().numpy()
        amplitude, phase = weighted_amplitude_phase(domain, spectra[i,:], mu_np)
        ax.plot(domain, mu)
        minus = (mu-sigma)
        plus = (mu+sigma)
        ax.fill_between(domain, minus, plus, color='grey')
        ax2 = ax.twinx()
        ax2.scatter(domain, spectra[i,:], color='k', s=10)
        ax.set_title("[%.2f, %.2f] : (%.2f, %.2f)"%(comps[i,0], comps[i,1], amplitude, phase))
        plt.savefig(eval_plot_dir+'%d.png'%(i))
        plt.close()
    
    return 

expt, comp_model, np_model = load_models_from_iteration(TOTAL_ITERATIONS)
plot_eval_comparision(wav_normalized, 
                      eval_comps.astype(np.double), 
                      eval_spectra, 
                      comp_model, 
                      np_model, 
                      "./"
                      )

accuracies = []
for i in range(1,TOTAL_ITERATIONS+1):
    expt, comp_model, np_model = load_models_from_iteration(i)
    eval_accuracy = get_accuracy(eval_comps.astype(np.double), 
                                 wav_normalized,
                                 eval_spectra,
                                 comp_model, 
                                 np_model
                                )
    accuracies.append(eval_accuracy)
    print("Iteration %d Evaluation Error : %2.4f (mean) %2.4f (std)"%(i, eval_accuracy.mean(), eval_accuracy.std()))

accuracies = np.asarray(accuracies)
fig, ax = plt.subplots()
x = np.arange(1,TOTAL_ITERATIONS+1)
y_mean = accuracies.mean(axis=1)
y_std = accuracies.std(axis=1)
ax.boxplot(accuracies.T)
plt.savefig("./eval_errorbar.png")
plt.close()
np.save("./eval_errors.npy", accuracies)