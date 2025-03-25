import os, sys, time, shutil, pdb, argparse, json
import numpy as np
import matplotlib.pyplot as plt

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)

from activephasemap.simulators import SAXSExperiment
from activephasemap.utils import *

parser = argparse.ArgumentParser(
                    prog='Train emulator of gold nanoparticle synthesis',
                    description='Perform a single iteration of active learning of Models 1 and 2',
                    epilog='...')
parser.add_argument('iteration', metavar='i', type=int, help='iterations number for the campaign')
args = parser.parse_args()
ITERATION = args.iteration # specify the current itereation number

# hyper-parameters
BATCH_SIZE = 6
N_INIT_POINTS = 12
DESIGN_SPACE_DIM = 2

EXPT_DATA_DIR = "./data/"
SAVE_DIR = './output/'
PLOT_DIR = './plots/'

if ITERATION==0:
    for direc in [EXPT_DATA_DIR, SAVE_DIR, PLOT_DIR]:
        if os.path.exists(direc):
            shutil.rmtree(direc)
        os.makedirs(direc)

""" Set up design space bounds """
design_space_bounds = [(0.005, 0.1), 
                       (0.005, 0.1),
                       (0.005, 0.15),
                       (0.1, 1.0)
                    ]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)


# Specify the pre-trained Neural Process model
best_np_config = { "r_dim":16 , "h_dim": 128, "z_dim": 8, "n_blocks": 5, "lr": 1e-3, "batch_size": 2}
np_model = NeuralProcess(best_np_config["r_dim"], 
                            best_np_config["z_dim"], 
                            best_np_config["h_dim"],
                            best_np_config["n_blocks"]
                            ).to(device)
model_path = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/SAXS/01b/model.pt"
np_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
np_model_args = {"num_iterations": 1000, 
                "verbose":100, 
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
          "np_model_args": np_model_args

}

if ITERATION == 0:
    init_x = initialize_points(bounds, N_INIT_POINTS, device)
    comps_init = init_x.detach().cpu().numpy()
    np.save(EXPT_DATA_DIR+'comps_0.npy', comps_init)
else: 
    expt = SAXSExperiment(design_space_bounds, EXPT_DATA_DIR)
    expt.read_iter_data(ITERATION)
    expt.generate()

    fig, ax = plt.subplots()
    expt.plot(ax)
    plt.savefig(PLOT_DIR+'train_spectra_%d.png'%ITERATION)
    plt.close()

    # obtain new set of compositions to synthesize
    result = run_iteration(expt, config)
    np.save(EXPT_DATA_DIR+'comps_%d.npy'%(ITERATION), result["comps_new"])

    plot_model_accuracy(expt, config, result)

