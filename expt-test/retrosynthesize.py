import os, sys, time, shutil, pdb, argparse,json
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
from torchmin import minimize_constr
# import Amplitude-Phase distance from https://github.com/kiranvad/Amplitude-Phase-Distance/tree/funcshape
from funcshape.functions import Function, SRSF, get_warping_function

from activephasemap.models.np import NeuralProcess, context_target_split
from activephasemap.utils.simulators import GNPPhases
from activephasemap.utils.settings import initialize_model 

gp_model_args = {"model":"gp", "num_epochs" : 1, "learning_rate" : 1e-3, "verbose": 1}
TRAINING_ITERATIONS = 50
DATA_DIR = './output'
ITERATION = 5
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]
DESIGN_SPACE_DIM = 2

design_space_bounds = [(0.0, 7.38), (0.0,7.27)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)

# Create a target spectrum
sim = GNPPhases("../AuNP/gold_nano_grid/")
target_comp = np.array([0.0, 4.4])
target = sim.simulate(target_comp) 
num_samples = target.shape[0]
xt = torch.from_numpy(sim.t).to(device).view(1, num_samples, 1)
yt = torch.from_numpy(target).to(device).view(1, num_samples, 1)

# Load GP and NP models and set them to evaluation mode
train_x = torch.load(DATA_DIR+'/train_x_%d.pt'%ITERATION, map_location=device)
train_y = torch.load(DATA_DIR+'/train_y_%d.pt'%ITERATION, map_location=device)
normalized_x = normalize(train_x, bounds).to(train_x)
print(normalized_x.max(), normalized_x.min())
GP = initialize_model(normalized_x, train_y, gp_model_args, DESIGN_SPACE_DIM, N_LATENT, device)
gp_state_dict = torch.load(DATA_DIR+'/gp_model_%d.pt'%ITERATION, map_location=device)
GP.load_state_dict(gp_state_dict)
GP.train(False)

NP = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
NP.load_state_dict(torch.load(DATA_DIR+'/np_model_%d.pt'%ITERATION, map_location=device)) 
NP.train(False)

def amplitude_phase_distance(t, f1, f2, **kwargs):
    """Define Amplitude-Phase distance as the loss function. 
    
    """
    t = (t-min(t))/(max(t)-min(t))
    f1 = Function(t, f1.reshape(-1,1))
    f2 = Function(t, f2.reshape(-1,1))

    with torch.no_grad():
        warping, network, error = get_warping_function(f1, f2, **kwargs) 

    q1, q2 = SRSF(f1), SRSF(f2)
    delta = q1.qx-q2.qx
    if delta.sum() == 0:
        amplitude, phase = torch.Tensor([0.0]).to(device), torch.Tensor([0.0]).to(device)
    else:
        network.project()
        gam_dev = network.derivative(t.unsqueeze(-1), h=None)
        q_gamma = q2(t)
        term1 = q1.qx.squeeze()
        term2 = q_gamma.squeeze() * torch.sqrt(gam_dev).squeeze()
        y = (term1 - term2) ** 2

        amplitude = torch.sqrt(torch.trapezoid(y, t))
        scaled_amplitude = amplitude/torch.sqrt(y.max())

        theta = torch.trapezoid(torch.sqrt(gam_dev).squeeze(), x=t)
        phase = torch.arccos(torch.clamp(theta, -1, 1))

    return scaled_amplitude, phase, [warping, network, error]

def simulator(c, mode="loss"):
    normalized_x = normalize(c, bounds)
    posterior = GP.posterior(normalized_x.reshape(1,-1)) 
    y_samples = []
    for _ in range(20):
        z = posterior.rsample().squeeze(0)
        y, _ = NP.xz_to_y(xt, z)
        y_samples.append(y)

    mu = torch.cat(y_samples).mean(dim=0, keepdim=True)
    sigma = torch.cat(y_samples).std(dim=0, keepdim=True)

    if not mode=="simulate":
        loss_type = mode.split("-")[1]
        if loss_type=="MSE":
            loss = torch.nn.functional.mse_loss(mu, yt)
        elif loss_type=="AP":
            optim_kwargs = {"n_iters":50, 
                            "n_basis":15, 
                            "n_layers":15,
                            "domain_type":"linear",
                            "basis_type":"palais",
                            "n_restarts":50,
                            "lr":1e-1,
                            "n_domain":num_samples,
                            "verbose":1
                            }
            
            amplitude, phase, _ = amplitude_phase_distance(xt.squeeze(), 
                                                        yt.squeeze(), 
                                                        mu.squeeze(),
                                                        **optim_kwargs
                                                        )
            loss = 0.5*(amplitude+phase)
            if torch.isnan(loss):
                torch.save([xt, yt, mu],"./check_apdist_nan.pt")
        else:
            raise RuntimeError("Loss type %s not recognized"%loss_type)
        print(c.data, normalized_x.data, loss.item())
        return loss
    else:

        return mu, sigma

C0 = draw_sobol_samples(bounds=bounds, n=1, q=1).to(device)

res = minimize_constr(lambda c : simulator(c, mode="loss-MSE"),
                      C0, 
                      bounds = dict(lb=bounds[0,:], ub=bounds[1,:]),
                      max_iter=TRAINING_ITERATIONS,
                      disp=0
                      )
print('final x: {}'.format(res.x))
print("Target composition : ", target_comp)
with torch.no_grad():
    fig, ax = plt.subplots()
    ax.plot(sim.t, target, label="Target")
    mu, sigma = simulator(res.x, mode="simulate")
    mu = mu.cpu().squeeze().numpy()
    sigma = sigma.cpu().squeeze().numpy()
    ax.plot(sim.t, mu, label="Best Estimate")
    ax.fill_between(sim.t,mu-sigma,mu+sigma,  color='grey', alpha=0.5, label="Uncertainity")
    ax.legend()
    plt.savefig("./plots/retrosynthesize_target.png")
    plt.show()

## A reason why this isn't working is because we have a stochastic model 
## so try some technqiues borrowed from this tutorial https://botorch.org/v/0.1.1/tutorials/optimize_stochastic