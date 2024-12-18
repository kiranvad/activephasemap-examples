import numpy as np 
import matplotlib.pyplot as plt 
import time, os, traceback, shutil, warnings, pickle, pdb
import ray 

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

from autophasemap import multi_kmeans_run, compute_BIC
from utils import AutoPhaseMapDataSet

# Specify variables
N_INIT_RUNS = 10
MAX_ITER = 50
VERBOSE = 3
GAMMA_GRID_N = 10
N_CLUSTERS = 3
DATA_DIR = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/11-2024-SeedAuNP"
SAVE_DIR = DATA_DIR+"/autophasemap/single_run/"
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

if not "ip_head" in os.environ:
    ray.init()
else:
    ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0],
        _redis_password=os.environ["redis_password"])

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

grid = np.load(DATA_DIR+"/paper/grid_data_20.npz")
grid_comps = grid["comps"]
grid_spectra = grid["spectra"][...,0]

t = np.linspace(0,1, grid_spectra.shape[1])

data = AutoPhaseMapDataSet(grid_comps, t, grid_spectra)
data.generate()

out, bic = multi_kmeans_run(N_INIT_RUNS, 
                            data, 
                            N_CLUSTERS, 
                            max_iter=MAX_ITER, 
                            verbose=VERBOSE, 
                            smoothen=True,
                            grid_dim = GAMMA_GRID_N
                            )

with open(SAVE_DIR+'/result_%d.pkl'%N_CLUSTERS, 'wb') as handle:
    pickle.dump(out._asdict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

# plot phase map and corresponding spectra
fig, axs = plot(data, out)
plt.savefig(SAVE_DIR+'/phase_map_%d.png'%N_CLUSTERS)
plt.close()

ray.shutdown()
