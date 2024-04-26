import numpy as np 
import matplotlib.pyplot as plt 
import time, os, traceback, shutil, warnings, pickle
import ray 

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from botorch.utils.transforms import normalize

from activephasemap.np.neural_process import NeuralProcess
from activephasemap.utils.settings import initialize_model
from activephasemap.test_functions.phasemaps import ExperimentalTestFunction
from activephasemap.utils.simulators import GNPPhases, UVVisExperiment
import matplotlib.ticker as ticker
from activephasemap.utils.settings import from_comp_to_spectrum, get_twod_grid, AutoPhaseMapDataSet
from activephasemap.utils.visuals import MinMaxScaler, scaled_tickformat, _inset_spectra 

from autophasemap import multi_kmeans_run, compute_BIC

# Specify variables
N_INIT_RUNS = 5
MAX_ITER = 20
VERBOSE = 3
GAMMA_GRID_N = 30
ACTIVEPHASEMAP_EXPT = "expt-test"

if ACTIVEPHASEMAP_EXPT=="expt-test":
    N_CLUSTERS = 4
    SAVE_DIR = "/mmfs1/home/kiranvad/kiranvad/papers/autophasemap/expts"+'/output/expt-test_single/'
    ITERATION = 8
    DATA_DIR = "/mmfs1/home/kiranvad/kiranvad/activephasemap-examples/expt-test"
    design_space_bounds = [(0.0, 7.38), (0.0,7.27)]
elif ACTIVEPHASEMAP_EXPT=="peptide_aunp_2D":
    N_CLUSTERS = 3
    SAVE_DIR = "/mmfs1/home/kiranvad/kiranvad/papers/autophasemap/expts"+'/output/peptide_aunp_2D_single/'
    ITERATION = 3
    DATA_DIR = "/mmfs1/home/kiranvad/kiranvad/040824/2D/"
    design_space_bounds = [(0.0, 87.0), (0.0,11.0)]

if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

if not "ip_head" in os.environ:
    ray.init()
else:
    ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0],
        _redis_password=os.environ["redis_password"])


grid_comps = np.load(DATA_DIR+"/grid/grid_comps.npy")
grid_spectra = np.load(DATA_DIR+"/grid/grid_spectra.npy")
t = np.linspace(0,1, grid_spectra.shape[1])

data = AutoPhaseMapDataSet(grid_comps,t, grid_spectra)
data.generate(process="normalize")

def plot(data, out):
    fig, axs = plt.subplots(1,N_CLUSTERS+1, figsize=(4*(N_CLUSTERS+1), 4))
    axs = axs.flatten() 
    n_clusters = len(out.templates)
    axs[n_clusters].scatter(data.C[:,0], data.C[:,1], c=out.delta_n)
    for k in range(n_clusters):
        Mk = np.argwhere(out.delta_n==k).squeeze()
        for i in Mk:
            axs[k].plot(data.t, out.fik_gam[i,k,:], color='grey')
        
        axs[k].plot(data.t, out.templates[k], lw=3.0, color='tab:red') 
        axs[k].axis('off')

    return fig, axs

out, bic = multi_kmeans_run(N_INIT_RUNS, 
                            data, 
                            N_CLUSTERS, 
                            max_iter=MAX_ITER, 
                            verbose=VERBOSE, 
                            smoothen=True,
                            grid_dim = GAMMA_GRID_N
                            )

with open(SAVE_DIR+'/result_%d.pkl'%N_CLUSTERS, 'wb') as handle:
    pickle.dump(out._asdict(), handle, 
        protocol=pickle.HIGHEST_PROTOCOL
        )

# plot phase map and corresponding spectra
fig, axs = plot(data, out)
plt.savefig(SAVE_DIR+'/phase_map_%d.png'%N_CLUSTERS)
plt.close()

ray.shutdown()
