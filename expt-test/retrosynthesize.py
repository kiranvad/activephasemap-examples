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

from activephasemap.np.neural_process import NeuralProcess 
from activephasemap.np.utils import context_target_split
from activephasemap.utils.simulators import GNPPhases
from activephasemap.utils.settings import initialize_model 

gp_model_args = {"model":"gp", "num_epochs" : 1, "learning_rate" : 1e-3, "verbose": 1}
np_model_args = {"num_iterations": 100, "verbose":True, "print_freq":100, "lr":5e-4}
LEARNING_RATE = 0.5
TRAINING_ITERATIONS = 50
DATA_DIR = './output'
ITERATION = 5
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
GP = initialize_model("gp", gp_model_args, DESIGN_SPACE_DIM, N_LATENT, device)
normalized_x = normalize(train_x, bounds).to(train_x)
gp_state_dict = torch.load(DATA_DIR+'/gp_model_%d.pt'%ITERATION, map_location=device)
GP.fit(normalized_x, train_y)
GP.load_state_dict(gp_state_dict)
GP.train(False)

NP = NeuralProcess(1, 1, 128, 2, 128).to(device)
NP.load_state_dict(torch.load(DATA_DIR+'/np_model_%d.pt'%ITERATION, map_location=device)) 
NP.train(False)

class Model(torch.nn.Module):
    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds
        self.constraint = Interval(bounds[0,:], bounds[1,:])
        init_param = draw_sobol_samples(bounds=bounds, n=1, q=1).to(device)
        self.param = torch.nn.Parameter(init_param, requires_grad=True)

    def forward(self, xt, yt, gp, np):
        normalized_c = torch.sigmoid(self.param)
        unnormalized_c = self.constraint.transform(normalized_c)
        print(self.param.data, normalized_c, unnormalized_c)
        posterior = gp.posterior(normalized_c.reshape(1,-1)) 
        y_samples = []
        for _ in range(250):
            z = posterior.rsample().squeeze(0)
            y, _ = np.xz_to_y(xt, z)
            y_samples.append(y)

        mu = torch.cat(y_samples).mean(dim=0, keepdim=True)
        sigma = torch.cat(y_samples).std(dim=0, keepdim=True)
        dist = Normal(mu, sigma)
        loss = dist.log_prob(yt).mean()

        return unnormalized_c, mu, sigma, loss         


model = Model(bounds).to(device)
model.train()

optim = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

for itr in range(TRAINING_ITERATIONS):
    optim.zero_grad()
    c_opt, mu, sigma, loss = model.forward(xt, yt, GP, NP)
    loss.backward()
    optim.step()
    print("Iteration %d, loss : %.4f and current best estimate : "%(itr, loss.item()), c_opt.data)

model.eval()
with torch.no_grad():
    fig, ax = plt.subplots()
    ax.plot(sim.t, target, label="Target")
    mu = mu.cpu().squeeze().numpy()
    sigma = sigma.cpu().squeeze().numpy()
    ax.plot(sim.t, mu, label="Best Estimate")
    ax.fill_between(sim.t,mu-sigma,mu+sigma,  color='grey', alpha=0.5, label="Uncertainity")
    ax.legend()
    plt.savefig("retrosynthesize_target.png")
    plt.show()