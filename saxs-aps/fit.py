import os, sys, time, shutil, pdb, argparse, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate 

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)

from activephasemap.simulators import SAXSExperiment
from activephasemap.utils import *

# hyper-parameters
BATCH_SIZE = 6
N_INIT_POINTS = 12
DESIGN_SPACE_DIM = 2
ITERATION = 1

EXPT_DATA_DIR = "./data/"
SAVE_DIR = './output/'
PLOT_DIR = './plots/'

""" Set up design space bounds """
design_space_bounds = [(0.0035, 0.04), # teof
                       (0.01, 0.08), # ammonia
                       (0.3, 1.0), # water
                       (0.2, 0.4), # ethanol
                       (10.0, 100.0), # ctab
                       (0.0, 300.0) # F127
                    ]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

# Specify the pre-trained Neural Process model
best_np_config = { "r_dim":128 , "h_dim": 128, "z_dim": 8, "n_blocks": 5, "lr": 1e-3, "batch_size": 2, "basis":"sine"}
np_model = NeuralProcess(best_np_config["r_dim"], 
                            best_np_config["z_dim"], 
                            best_np_config["h_dim"],
                            best_np_config["n_blocks"],
                            best_np_config["basis"]
                            ).to(device)
model_path = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/SAXS/01b/model.pt"
np_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
np_model_args = {"num_iterations": 500, 
                "verbose":50, 
                "lr":best_np_config["lr"], 
                "batch_size": best_np_config["batch_size"]
                }

# Specify parameters for composition-latent model
xgb_model_args = {"objective": "reg:squarederror",
                  "eval_metric": "rmse"
                  }

config = {"iteration" : ITERATION,
          "expt_data_dir" : EXPT_DATA_DIR,
          "save_dir" : SAVE_DIR,
          "plot_dir" : PLOT_DIR,
          "batch_size" : BATCH_SIZE,
          "dimension" : DESIGN_SPACE_DIM,
          "bounds" : design_space_bounds,
          "np_model" : np_model,
          "n_z_draws" : 256,
          "best_np_config":best_np_config,
          "xgb_model_args":xgb_model_args,
          "acqf_n_restarts":4,
          "acqf_n_iterations":10,
          "np_model_args": np_model_args

}

class SAXSAPS(SAXSExperiment):
    def __init__(self, bounds, direc):
        super().__init__(bounds, direc)
        self.n_domain = 100
        with open(self.dir+'/aps.pkl', 'rb') as handle:
            self.data = pickle.load(handle) 

    def spline_interpolate(self, q, Iq):
        q_grid = np.linspace(np.log10(min(q)), 
                             np.log10(max(q)), 
                             self.n_domain
                             )
        spline = interpolate.splrep(np.log10(q), np.log10(np.abs(Iq)), s=0)
        I_grid = interpolate.splev(q_grid, spline, der=0)

        return q_grid, I_grid 

    def generate(self):
        C, F, T = [], [], []
        for key in self.data.keys():
            c = [*self.data[key]["composition"].values()]
            saxs = self.data[key]["saxs_data"][0]
            q = saxs["q"].to_numpy()
            Iq = saxs["I"].to_numpy()
            q_grid, Iq_grid = self.spline_interpolate(q, Iq)
            C.append(c)
            T.append(q_grid)
            F.append(Iq_grid)

        self.t = np.asarray(T)
        self.spectra_normalized = np.asarray(F)
        self.comps = np.asarray(C)
    
        print("Composition shape :", self.comps.shape)
        print("Functional data domain shape : ", self.t.shape)
        print("Functional data co-domain shape : ", self.spectra_normalized.shape)

    def plot(self, ax):
        for ti, si in zip(self.t, self.spectra_normalized):
            ax.plot(ti, si, color="tab:blue", alpha=0.5)
        ax.set_xlabel(r"$\log(q)$")
        ax.set_ylabel(r"$\log(I(q))$")

        return 

expt = SAXSAPS(design_space_bounds, EXPT_DATA_DIR)
expt.generate()

fig, ax = plt.subplots()
expt.plot(ax)
plt.savefig(PLOT_DIR+'train_spectra_%d.png'%ITERATION)
plt.close()

# obtain new set of compositions to synthesize
result = run_iteration(expt, config)
# np.save(EXPT_DATA_DIR+'comps_%d.npy'%(ITERATION), result["comps_new"])

plot_model_accuracy(expt, config, result)

# Plot grid filling on teos vs ethanol
with torch.no_grad():
    fig, axs = plt.subplots(1,2, figsize=(5*2, 5))
    bounds_np_2d = np.asarray(design_space_bounds)[[0, 3],:].T
    scaler_x = MinMaxScaler(bounds_np_2d[0,0], bounds_np_2d[1,0])
    scaler_y = MinMaxScaler(bounds_np_2d[0,1], bounds_np_2d[1,1])

    # (left) Collected data on 2D but others randomized
    ax = axs[0]
    for i in range(expt.comps.shape[0]):
        norm_ci = np.array([scaler_x.transform(expt.comps[i,0]), 
                            scaler_y.transform(expt.comps[i,3])
                            ]
                        )

        inset_spectra(norm_ci, 
                    expt.t[i,:], 
                    expt.spectra_normalized[i,:], 
                    [], 
                    ax, 
                    show_sigma=False,
                    color="k",
                    lw=0.5
                    )
    ax.set_xlabel('TEOS')
    ax.set_ylabel('Ethanol')

    ax = axs[1]
    ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
    ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    C_grid = get_twod_grid(10, bounds_np_2d)

    for ci in C_grid:
        t_star = np.linspace(expt.t.min(), expt.t.max(), expt.n_domain)
        n_rnd_samples = 10
        ci_expand = np.zeros((n_rnd_samples, len(design_space_bounds)))
        ci_rnd = initialize_points(bounds, n_rnd_samples, device).detach().cpu().numpy()
        ci_expand[:,[0,3]] = ci
        ci_expand[:,[1,2,4]] = ci_rnd[:,[1,2,4]] 
        for i in range(n_rnd_samples):
            mu, _ = from_comp_to_spectrum(t_star,ci_expand[i,:], result["comp_model"], result["np_model"])
            mu_ = mu.cpu().squeeze().numpy()
            norm_ci = np.array([scaler_x.transform(ci_expand[i,0]), scaler_y.transform(ci_expand[i,3])])
            inset_spectra(norm_ci, t_star, mu_, [], ax, show_sigma=False, color="k", alpha=0.5)
    ax.set_xlabel('TEOS')
    ax.set_ylabel('Ethanol')
    plt.savefig(PLOT_DIR+"expt_model_compare.png")
    plt.close()