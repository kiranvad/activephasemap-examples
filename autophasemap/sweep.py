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
from activephasemap.utils.simulators import UVVisExperiment
import matplotlib.ticker as ticker
from activephasemap.utils.settings import from_comp_to_spectrum, get_twod_grid, AutoPhaseMapDataSet
from activephasemap.utils.visuals import MinMaxScaler, scaled_tickformat, _inset_spectra 

from autophasemap import multi_kmeans_run, compute_BIC, BaseDataSet

# Specify variables
N_CLUSTERS_MIN = 1
N_CLUSTERS_MAX = 5
N_INIT_RUNS = 5
MAX_ITER = 20
VERBOSE = 3
GAMMA_GRID_N = 7
ACTIVEPHASEMAP_EXPT = "expt-test"

if ACTIVEPHASEMAP_EXPT=="expt-test":
    SAVE_DIR = "/mmfs1/home/kiranvad/kiranvad/papers/autophasemap/expts"+'/output/expt_test_sweep/'
    ITERATION = 8
    DATA_DIR = "/mmfs1/home/kiranvad/kiranvad/activephasemap-examples/expt-test"
    design_space_bounds = [(0.0, 7.38), (0.0,7.27)]
elif ACTIVEPHASEMAP_EXPT=="peptide_aunp_2D":
    SAVE_DIR = "/mmfs1/home/kiranvad/kiranvad/papers/autophasemap/expts"+'/output/peptide_aunp_2D_sweep/'
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
data.generate(process=None)

def plot(data, out):
    fig, axs = plt.subplots(1,N_CLUSTERS_MAX+1, figsize=(4*(N_CLUSTERS_MAX+1), 4))
    fig.subplots_adjust(wspace=0.5)
    axs[-1].scatter(data.C[:,0], data.C[:,1], c=out.delta_n)
    for k in range(len(out.templates)):
        Mk = np.argwhere(out.delta_n==k).squeeze()
        for i in Mk:
            axs[k].plot(data.t, out.fik_gam[i,k,:], color='grey')
        
        axs[k].plot(data.t, out.templates[k], lw=3.0, color='tab:red') 

    return fig, axs

BIC_LIST = []
for N_CLUSTERS in np.arange(N_CLUSTERS_MIN,N_CLUSTERS_MAX+1):
    print("Running autophasemap with %d clusters..."%N_CLUSTERS)
    if N_CLUSTERS==1:
        N_INIT_RUNS_ARG = 1
    else:
        N_INIT_RUNS_ARG = N_INIT_RUNS
    out, bic = multi_kmeans_run(N_INIT_RUNS_ARG, 
                                data, 
                                N_CLUSTERS, 
                                max_iter=MAX_ITER, 
                                verbose=VERBOSE, 
                                smoothen=True,
                                grid_dim = GAMMA_GRID_N
                                )
    BIC_LIST.append(bic)

    with open(SAVE_DIR+'/result_%d.pkl'%N_CLUSTERS, 'wb') as handle:
        pickle.dump(out._asdict(), handle, 
            protocol=pickle.HIGHEST_PROTOCOL
            )

    # plot phase map and corresponding spectra
    fig, ax = plot(data, out)
    fig.suptitle("Average BIC : %2.4f"%np.asarray(bic).mean())
    plt.savefig(SAVE_DIR+'/phase_map_%d.png'%N_CLUSTERS)
    plt.close()

ray.shutdown()

np.save(SAVE_DIR+"/bic.npy", np.asarray(BIC_LIST))
fig, ax = plt.subplots()
ax.errorbar(np.arange(N_CLUSTERS_MIN, N_CLUSTERS_MAX+1), 
            np.asarray(BIC_LIST).mean(axis=1),
            np.asarray(BIC_LIST).std(axis=1)
)
plt.savefig(SAVE_DIR+'/bic.png')
plt.close()
