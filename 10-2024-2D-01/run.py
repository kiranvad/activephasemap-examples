import os, sys, time, shutil, pdb, argparse, json
import numpy as np
import matplotlib.pyplot as plt

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)

from activephasemap.simulators import UVVisExperiment
from activephasemap.utils import *
# sys.path.append("/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/08-2024-2D/")
# from utils import *

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
design_space_bounds = [(0.0, 87.0), (0.0,11.0)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

config = {"iteration" : ITERATION,
          "expt_data_dir" : EXPT_DATA_DIR,
          "save_dir" : SAVE_DIR,
          "plot_dir" : PLOT_DIR,
          "batch_size" : BATCH_SIZE,
          "dimension" : DESIGN_SPACE_DIM,
          "bounds" : design_space_bounds
}

# Set up a synthetic data emulating an experiment
if ITERATION == 0:
    init_x = initialize_points(bounds, N_INIT_POINTS, device)
    comps_init = init_x.detach().cpu().numpy()
    np.save(EXPT_DATA_DIR+'comps_0.npy', comps_init)
else: 
    expt = UVVisExperiment(design_space_bounds, EXPT_DATA_DIR)
    expt.read_iter_data(ITERATION)
    expt.generate(use_spline=True)

    fig, ax = plt.subplots()
    expt.plot(ax, design_space_bounds)
    plt.savefig(PLOT_DIR+'train_spectra_%d.png'%ITERATION)
    plt.close()

    # obtain new set of compositions to synthesize
    result = run_iteration(expt, config)
    np.save(EXPT_DATA_DIR+'comps_%d.npy'%(ITERATION), result["comps_new"])

    plot_model_accuracy(expt, config, result)

    plot_iteration(expt, config, result)
    plt.savefig(PLOT_DIR+'itr_%d.png'%ITERATION)
    plt.close()
