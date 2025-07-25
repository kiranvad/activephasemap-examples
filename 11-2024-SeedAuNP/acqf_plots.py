import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colormaps 
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
pyplot_style ={"text.usetex": True,
               "text.latex.preamble": r"\usepackage{amsfonts}\usepackage[version=4]{mhchem}",
               "axes.spines.right" : False,
               "axes.spines.top" : False,
               "font.size": 22,
               "savefig.dpi": 600,
               "savefig.bbox": 'tight',
              } 

import torch
import argparse, json, glob, os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from activephasemap.models.np import NeuralProcess
from activephasemap.models.acquisition import XGBUncertainity
from activephasemap.simulators import UVVisExperiment
from activephasemap.models.xgb import XGBoost
from activephasemap.utils import *

import importlib.resources as pkg_resources

parser = argparse.ArgumentParser(
                    prog='Train emulator of gold nanoparticle synthesis',
                    description='Perform a single iteration of active learning of Models 1 and 2',
                    epilog='...')
parser.add_argument('iteration', metavar='i', type=int, help='iterations number for the campaign')
args = parser.parse_args()
ITERATION = args.iteration # specify the current itereation number

DATA_DIR = "./output/"
with pkg_resources.open_text("activephasemap.pretrained", "best_config.json") as file:    
    best_np_config = json.load(file)
N_LATENT = best_np_config["z_dim"]
TOTAL_ITERATIONS  = len(glob.glob(DATA_DIR+"comp_model_*.json"))

# Load trained NP model for p(y|z)
np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
np_model.load_state_dict(torch.load(DATA_DIR+'np_model_%d.pt'%(ITERATION), map_location=device, weights_only=True))

# Load trained composition to latent model for p(z|c)
xgb_model_args = {"objective": "reg:squarederror",
                  "max_depth": 3,
                  "eta": 0.1,
                  "eval_metric": "rmse"
                  }
comp_model = XGBoost(xgb_model_args)
comp_model.load(DATA_DIR+"comp_model_%d.json"%ITERATION)

design_space_bounds = [(0.0, 35.0), (0.0, 35.0)]

# Create the experiment class to load all the data obtained so far
expt = UVVisExperiment(design_space_bounds, "./data/")
expt.read_iter_data(ITERATION)
expt.generate(use_spline=True)
bounds_np = expt.bounds.cpu().numpy()

# load composition data requested for next iteration
comps_new = np.load("./data/comps_%d.npy"%(ITERATION+1))

acqf = XGBUncertainity(expt, expt.bounds, np_model, comp_model)
C_grid = get_twod_grid(15, bounds_np)
with torch.no_grad():
     C_grid_ = torch.tensor(C_grid).reshape(len(C_grid),1,2).to(device)
     rx_train_, rx_, sigma_x_ = acqf(C_grid_, return_rx_sigma=True)
     rx_train_norm = acqf._min_max_normalize(rx_train_)

# Plot acquisiton function
rx_train = rx_train_norm.squeeze().cpu().numpy()
rx = rx_.squeeze().cpu().numpy()
sigma_x = sigma_x_.squeeze().cpu().numpy()

def prettify_axis(ax):
    """Code to modify design space axis

    This is an auxilary function to modfiy the axis
    from volumes to concentrations as well as make them
    look pretty with axis lines and tick labels.

    Parameters
    ----------
    ax : pyplot.axis
        axis object to plot on

    Returns
    -------
    pyplot.axis
        axis object modified tick labels and pretty-fied
    """
    ax.set_xlabel(r"Silver Nitrate ($10^{-4}$ M)")
    ax.set_ylabel(r"Ascorbic Acid ($10^{-3}$ M)")
    ax.set_xlim(*design_space_bounds[0])
    ax.set_ylim(*design_space_bounds[1])
    ax.xaxis.set_major_locator(ticker.LinearLocator(3))
    ax.yaxis.set_major_locator(ticker.LinearLocator(5))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    SN_labels = [item.get_text() for item in ax.get_xticklabels()]
    # convert volume to concentration using c1v1 = c2v2
    SN_vol_to_conc = lambda v : (6.4*v)/(300)
    SN_conc_labels = []
    for l in SN_labels:
        SN_conc_labels.append("%.2f"%SN_vol_to_conc(float(l)))
    ax.set_xticklabels(SN_conc_labels)

    AA_labels = [item.get_text() for item in ax.get_yticklabels()]
    AA_vol_to_conc = lambda v : (6.3*v)/(300)
    AA_conc_labels = []
    for l in AA_labels:
        AA_conc_labels.append("%.2f"%AA_vol_to_conc(float(l)))
    ax.set_yticklabels(AA_conc_labels)

    return ax

# Plot r(x) and its extrpolation over the design space
with plt.style.context(pyplot_style):
    fig, ax = plt.subplots(figsize=(5,5))
    norm = Normalize(vmin=0, vmax = 0.5)
    cmap = colormaps["magma"]
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    ax.tricontourf(C_grid[:,0], C_grid[:,1], rx, cmap=cmap, norm=norm)
    ax.scatter(expt.comps[:,0], expt.comps[:,1], s=rx_train*100, color='w')
    size_legend_values = [20, 50, 100]
    handles = [
        ax.scatter([], [], s=size, color='w',
                label=fr'${int(size)}\%$')
        for size in size_legend_values
    ]

    legend = ax.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.35),
        ncol=len(handles),
        title=r"$r_{i}$",
        frameon=True,
        facecolor='k',
        framealpha = 0.5,
        fancybox = True,
        fontsize=18,    
        handletextpad=0.4,     # ↓ space between symbol and label
        columnspacing=1.0,     # ↓ space between columns
        handlelength=1.2       # ↓ length of marker symbol
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel(r"$\alpha r(c)$")
    prettify_axis(ax)
    plt.savefig("./misc/acqf_plots/%dA.png"%ITERATION)
    plt.close()

with plt.style.context(pyplot_style):
    fig, ax = plt.subplots(figsize=(5,5))
    norm = Normalize(vmin=0, vmax = 0.5)
    cmap = colormaps["magma"]
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    ax.tricontourf(C_grid[:,0], C_grid[:,1], sigma_x, cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel(r"$(1-\alpha) \sigma(x)$")
    prettify_axis(ax) 
    plt.savefig("./misc/acqf_plots/%dB.png"%ITERATION)
    plt.close()

with plt.style.context(pyplot_style):
    fig, ax = plt.subplots(figsize=(5,5))
    norm = Normalize(vmin=0, vmax = 1.0)
    cmap = colormaps["magma"]
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    ax.tricontourf(C_grid[:,0], C_grid[:,1], rx+sigma_x, cmap=cmap, norm=norm)
    ax.scatter(comps_new[:,0], comps_new[:,1], marker='x', color='w')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel(r"$\alpha r_(x) + (1-\alpha) \sigma(x)$")
    prettify_axis(ax) 
    plt.savefig("./misc/acqf_plots/%dC.png"%ITERATION)
    plt.close()
