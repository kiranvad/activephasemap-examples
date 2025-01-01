import torch
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import pdb, argparse, json, glob, pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

from activephasemap.models.np import NeuralProcess
from activephasemap.utils import *

ITERATION = 14

DATA_DIR = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/11-2024-SeedAuNP/"
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap/activephasemap/pretrained/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]
TOTAL_ITERATIONS  = len(glob.glob(DATA_DIR+"/output/comp_model_*.json"))

# Load finetuned NP model from last iteration
finetuned = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
finetuned.load_state_dict(torch.load(DATA_DIR+'/output/np_model_%d.pt'%(ITERATION), map_location=device, weights_only=True))

# Load pre-trained NP model
pretrained = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
pretrained.load_state_dict(torch.load("/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap/activephasemap/pretrained/np_model.pt", 
                                      map_location=device, 
                                      weights_only=True
                                      )
                           )

x_target = torch.linspace(0, 1, 100).reshape(1,100,1).to(device)
z_sample = torch.randn((100, N_LATENT))
mu_ft_samples, mu_pt_samples = [],[]
fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
with torch.no_grad():
    for zi in z_sample:
        mu_ft, _ = finetuned.xz_to_y(x_target, zi.to(device))
        mu_pt, _ = pretrained.xz_to_y(x_target, zi.to(device))
        axs[0].plot(x_target.cpu().numpy()[0], mu_pt.detach().cpu().numpy()[0], c='b', alpha=0.5)
        axs[1].plot(x_target.cpu().numpy()[0], mu_ft.detach().cpu().numpy()[0], c='b', alpha=0.5)
        mu_ft_samples.append(mu_ft.detach().cpu().numpy()[0])
        mu_pt_samples.append(mu_pt.detach().cpu().numpy()[0])
plt.savefig("./np_samples.png")
plt.close()

np.savez("./finetuned_samples.npz", finetuned=np.asarray(mu_ft_samples), pretrained=np.asarray(mu_pt_samples))



