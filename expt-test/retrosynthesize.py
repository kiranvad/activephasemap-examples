import os, sys, time, shutil, pdb, argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from gpytorch.constraints import Interval
from botorch.utils.sampling import draw_sobol_samples
torch.set_default_dtype(torch.double)
from botorch.utils.transforms import normalize, unnormalize
from torch.distributions import Normal

# import minimize function from https://github.com/rfeinman/pytorch-minimize
from torchmin import minimize
# import Amplitude-Phase distance from https://github.com/kiranvad/Amplitude-Phase-Distance/tree/funcshape
from apdist.torch import AmplitudePhaseDistance as apdist

from activephasemap.models.np.neural_process import NeuralProcess 
from activephasemap.models.np.utils import context_target_split
from activephasemap.utils.simulators import GNPPhases
from activephasemap.utils.settings import initialize_model 

gp_model_args = {"model":"gp", "num_epochs" : 1, "learning_rate" : 1e-3, "verbose": 1}
np_model_args = {"num_iterations": 100, "verbose":True, "print_freq":100, "lr":5e-4}
TRAINING_ITERATIONS = 50
DATA_DIR = './output'
ITERATION = 8
N_LATENT = 2
DESIGN_SPACE_DIM = 2

design_space_bounds = [(0.0, 7.38), (0.0,7.27)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)
sim = GNPPhases("../AuNP/gold_nano_grid/") 
target = sim.simulate(np.array([0.0, 4.4])) 
num_samples = target.shape[0]
xt = torch.from_numpy(sim.t).to(device).view(1, num_samples, 1)
yt = torch.from_numpy(target).to(device).view(1, num_samples, 1) 
train_x = torch.load(DATA_DIR+'/train_x_%d.pt'%ITERATION, map_location=device)
train_y = torch.load(DATA_DIR+'/train_y_%d.pt'%ITERATION, map_location=device)
normalized_x = normalize(train_x, bounds).to(train_x)
GP = initialize_model(normalized_x, train_y, gp_model_args, DESIGN_SPACE_DIM, N_LATENT, device)
gp_state_dict = torch.load(DATA_DIR+'/gp_model_%d.pt'%ITERATION, map_location=device)
GP.load_state_dict(gp_state_dict)
GP.train(False)

NP = NeuralProcess(1, 1, 128, 2, 128).to(device)
NP.load_state_dict(torch.load(DATA_DIR+'/np_model_%d.pt'%ITERATION, map_location=device)) 
NP.train(False)

def simulator(c, mode=None):
    normalized_x = normalize(c, bounds)
    posterior = GP.posterior(normalized_x.reshape(1,-1)) 
    y_samples = []
    for _ in range(250):
        z = posterior.rsample().squeeze(0)
        y, _ = NP.xz_to_y(xt, z)
        y_samples.append(y)

    mu = torch.cat(y_samples).mean(dim=0, keepdim=True)
    sigma = torch.cat(y_samples).std(dim=0, keepdim=True)

    if mode is None:
        loss = torch.nn.functional.mse_loss(mu, yt)
        # optim_kwargs = {"n_iters":50, 
        #                 "n_basis":15, 
        #                 "n_layers":15,
        #                 "domain_type":"linear",
        #                 "basis_type":"palais",
        #                 "n_restarts":50,
        #                 "lr":1e-1,
        #                 "n_domain":num_samples
        #                 }
        
        # amplitude, phase, _ = apdist(xt.squeeze(), 
        #                              yt.squeeze(), 
        #                              mu.squeeze(), 
        #                              **optim_kwargs
        #                              )
        # loss = amplitude+phase
        if torch.isnan(loss):
            torch.save([xt, yt, mu],"./check_apdist_nan.pt")

        print("Current amplitude-phase distance : ", loss.item())

        return loss
    else:

        return mu, sigma

C0 = draw_sobol_samples(bounds=bounds, n=1, q=1).to(device)

res = minimize(
    simulator, C0, 
    method='newton-cg', 
    options=dict(line_search='strong-wolfe'),
    max_iter=TRAINING_ITERATIONS,
    disp=2
)
print('final x: {}'.format(res.x))

with torch.no_grad():
    fig, ax = plt.subplots()
    ax.plot(sim.t, target, label="Target")
    mu, sigma = simulator(res.x, mode="compute")
    mu = mu.cpu().squeeze().numpy()
    sigma = sigma.cpu().squeeze().numpy()
    ax.plot(sim.t, mu, label="Best Estimate")
    ax.fill_between(sim.t,mu-sigma,mu+sigma,  color='grey', alpha=0.5, label="Uncertainity")
    ax.legend()
    plt.savefig("./plots/retrosynthesize_target.png")
    plt.show()