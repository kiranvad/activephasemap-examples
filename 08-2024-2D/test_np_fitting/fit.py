from activephasemap.models.np import NeuralProcess
from activephasemap.models.utils import finetune_neural_process
from activephasemap.utils.simulators import UVVisExperiment
import os, sys, time, shutil, pdb, argparse, json, glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)


PRETRAIN_LOC = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/test_np_new_api/model.pt"
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]
np_model_args = {"num_iterations": 100, 
                 "verbose":100, 
                 "lr":best_np_config["lr"], 
                 "batch_size": best_np_config["batch_size"]
                 }

EXPT_DATA_DIR = "../data/" 
ITERATION = len(glob.glob("../output/comp_model_*.pt"))

design_space_bounds = [(0.0, 87.0), (0.0,11.0)]
expt = UVVisExperiment(design_space_bounds, EXPT_DATA_DIR)
expt.read_iter_data(ITERATION)
expt.generate(use_spline=True)
comps_all = expt.comps 
spectra_all = expt.spectra_normalized # use normalized spectra
print('Data shapes : ', comps_all.shape, spectra_all.shape)
n_domain = 100
x_target = torch.linspace(0, 1, n_domain).reshape(1,n_domain,1).to(device) 

# Specify the Neural Process model
np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
np_model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device, weights_only=True))

def plot_samples(model, x_target, num_samples=100):
    fig, ax = plt.subplots()
    z_samples = -5.0 + 10.0*torch.randn((20, N_LATENT)).to(device)
    with torch.no_grad():
        for zi in z_samples:
            mu, _ = model.xz_to_y(x_target, zi.to(device))
            ax.plot(x_target.cpu().numpy()[0], 
                    mu.detach().cpu().numpy()[0], 
                    c='b', 
                    alpha=0.5
                    )

    return 

def plot_posterior_samples(model, x_target, expt):
    fig, axs = plt.subplots(2,5, figsize=(4*5, 4*2))
    rids = torch.randint(0, expt.comps.shape[0], (5,))
    for i in range(5):
        inds = torch.randint(0, n_domain, (int(0.8*n_domain),))
        x_context = torch.from_numpy(expt.t[inds]).view(1,len(inds),1).to(device)
        y_context = torch.from_numpy(expt.spectra_normalized[rids[i],inds]).view(1,len(inds),1).to(device)
        z_values = torch.zeros((200, N_LATENT)).to(device)
        with torch.no_grad():
            for j in range(200):
                mu_context, sigma_context = np_model.xy_to_mu_sigma(x_context, y_context)
                q_context = torch.distributions.Normal(mu_context, sigma_context)
                z_values[j,:] = q_context.rsample()
                y_pred_mu, y_pred_sigma = np_model.xz_to_y(x_target, z_values[j,:])
                p_y_pred = torch.distributions.Normal(y_pred_mu, y_pred_sigma)

                # Extract mean of distribution
                mu = p_y_pred.loc.detach()
                axs[0,i].plot(x_target.squeeze().cpu().numpy(), 
                              mu.squeeze().cpu().numpy(), 
                              alpha=0.05, c='b'
                              )


            axs[0,i].scatter(x_context.squeeze().cpu().numpy(), 
                           y_context.squeeze().cpu().numpy(), 
                           c='tab:red'
                           )
            axs[0,i].plot(expt.t, expt.spectra_normalized[rids[i],:] ,c='tab:red')
            axs[0,i].set_title("[%d] : [%d, %.2f]"%(rids[i], 
                                                  expt.comps[rids[i],0], 
                                                  expt.comps[rids[i],1]
                                                  )
                            )
            axs[1,i].violinplot(z_values.cpu().numpy(), showmeans=True)

    return fig, axs

# Plot samples before finetuning
with torch.no_grad():
    plot_samples(np_model, x_target)
    plt.savefig('./finetune_before.png')
    plt.close()

print("Finetuning Neural Process model: ")
np_model, np_loss = finetune_neural_process(expt.t, spectra_all, np_model, **np_model_args)

with torch.no_grad():
    plot_samples(np_model, x_target)
    plt.savefig('./finetune_after.png')
    plt.close()

    # Plot curve fitting-like samples from posteriors
    plot_posterior_samples(np_model, x_target, expt)
    plt.savefig('./finetuned_posterior.png')
    plt.close()