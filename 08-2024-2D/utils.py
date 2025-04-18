import os, sys, time, shutil, pdb, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
start = time.time()

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
end = time.time()

print("Importing torch took : ", end-start)

from activephasemap.models.np import NeuralProcess, context_target_split
from activephasemap.models.utils import finetune_neural_process
from activephasemap.models.mlp import MLP
from activephasemap.acquisition import CompositeModelUncertainity
from activephasemap.simulators import UVVisExperiment

RNG = np.random.default_rng()

import seaborn as sns 
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps 
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

PRETRAIN_LOC = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/test_np_new_api/model.pt"

with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]
N_Z_DRAWS = 256
mlp_model_args = {"num_epochs" : 1000, 
                 "learning_rate" : 1e-3, 
                 "verbose": 100,
                 }

np_model_args = {"num_iterations": 1000, 
                 "verbose":250, 
                 "lr":best_np_config["lr"], 
                 "batch_size": best_np_config["batch_size"]
                 }

def get_twod_grid(n_grid, bounds):
    x = np.linspace(bounds[0,0],bounds[1,0], n_grid)
    y = np.linspace(bounds[0,1],bounds[1,1], n_grid)
    X,Y = np.meshgrid(x,y)
    points = np.vstack([X.ravel(), Y.ravel()]).T 

    return points 

def _inset_spectra(c, t, mu, sigma, ax, show_sigma=False, **kwargs):
        loc_ax = ax.transLimits.transform(c)
        ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
        ins_ax.plot(t, mu, **kwargs)
        if show_sigma:
            ins_ax.fill_between(t,mu-sigma, mu+sigma,
            color='grey')
        ins_ax.axis('off')
        ins_ax.set_ylim([0,1.8])
        
        return 

class MinMaxScaler:
    def __init__(self, min, max):
        self.min = min 
        self.max = max 
        self.range = max-min

    def transform(self, x):
        return (x-self.min)/self.range
    
    def inverse(self, xt):
        return (self.range*xt)+self.min

def scaled_tickformat(scaler, x, pos):
    return '%.1f'%scaler.inverse(x)

@torch.no_grad
def print_matrix(A):
    A = pd.DataFrame(A.cpu().numpy())
    A.columns = ['']*A.shape[1]
    print(A.to_string())

""" Helper functions """
@torch.no_grad()
def featurize_spectra(np_model, comps_all, spectra_all):
    """ Obtain latent space embedding from spectra.
    """
    num_samples, n_domain = spectra_all.shape
    spectra = torch.zeros((num_samples, n_domain)).to(device)
    for i, si in enumerate(spectra_all):
        spectra[i] = torch.tensor(si).to(device)
    t = torch.linspace(0, 1, n_domain).repeat(num_samples, 1).to(device)

    inds = torch.randint(0, n_domain, (int(0.95*n_domain),))
    x_context = t[:,inds].unsqueeze(-1) 
    y_context = spectra[:,inds].unsqueeze(-1)
    mu_context, sigma_context = np_model.xy_to_mu_sigma(x_context, y_context)
    q_context = torch.distributions.Normal(mu_context, sigma_context)
        
    train_y = q_context.rsample(torch.Size([N_Z_DRAWS]))
    
    z_mean = train_y.mean(dim=0)
    z_std = train_y.std(dim=0)
    train_x = torch.from_numpy(comps_all).to(device)

    return train_x, z_mean, z_std

def run_iteration(expt, config):
    """ Perform a single iteration of active phasemapping.

    helper function to run a single iteration given 
    all the compositions and spectra obtained so far. 
    """
    # assemble data for surrogate model training  
    comps_all = expt.comps 
    spectra_all = expt.spectra_normalized # use normalized spectra
    print('Data shapes : ', comps_all.shape, spectra_all.shape)
    result = {"comps_all" : comps_all,
              "spectra_all" : spectra_all,
    }

    # Specify the Neural Process model
    np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
    np_model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device, weights_only=True))

    print("Finetuning Neural Process model: ")
    np_model, np_loss = finetune_neural_process(expt.t, spectra_all, np_model, **np_model_args)
    torch.save(np_model.state_dict(), config["save_dir"]+'np_model_%d.pt'%config["iteration"])
    np.save(config["save_dir"]+'np_loss_%d.npy'%config["iteration"], np_loss)
    result["np_model"] = np_model
    result["np_loss"] = np_loss

    train_x, train_z_mean, train_z_std = featurize_spectra(np_model, comps_all, spectra_all)
    
    comp_model = MLP(train_x, train_z_mean, train_z_std, **mlp_model_args)
    print("Training comosition model p(z|C): ")
    comp_train_loss, comp_eval_loss = comp_model.fit(use_early_stoping=True)

    np.save(config["save_dir"]+'comp_train_loss_%d.npy'%config["iteration"], comp_train_loss)
    np.save(config["save_dir"]+'comp_eval_loss_%d.npy'%config["iteration"], comp_eval_loss)
    torch.save(comp_model.state_dict(), config["save_dir"]+'comp_model_%d.pt'%config["iteration"])
    result["comp_model"] = comp_model
    result["comp_train_loss"] = comp_train_loss
    result["comp_eval_loss"] = comp_eval_loss

    print("Collecting next data points to sample by acqusition optimization...")
    bounds = torch.tensor(config["bounds"]).transpose(-1, -2).to(device)
    acqf = CompositeModelUncertainity(expt.t, bounds, np_model, comp_model)
    new_x = acqf.optimize(config["batch_size"])
    print_matrix(new_x)

    torch.save(train_x.cpu(), config["save_dir"]+"train_x_%d.pt" %config["iteration"])
    torch.save(train_z_mean.cpu(), config["save_dir"]+"train_z_mean_%d.pt" %config["iteration"])
    torch.save(train_z_std.cpu(), config["save_dir"]+"train_z_std_%d.pt" %config["iteration"])

    result["acqf"] = acqf
    result["comps_new"] = new_x.cpu().numpy()
    result["train_x"] = train_x
    result["train_z_mean"] = train_z_mean 
    result["train_z_std"] = train_z_std

    return result

def from_comp_to_spectrum(t, c, comp_model, np_model, return_mlp_outputs=False):
    ci = torch.tensor(c).to(device)
    z_mu, z_std = comp_model.mlp(ci)
    z_dist = torch.distributions.Normal(z_mu, z_std)
    z = z_dist.sample(torch.Size([100]))
    t = torch.from_numpy(t).repeat(100, 1, 1).to(device)
    t = torch.swapaxes(t, 1, 2)
    y_samples, _ = np_model.xz_to_y(t, z)

    mean_pred = y_samples.mean(dim=0, keepdim=True)
    sigma_pred = y_samples.std(dim=0, keepdim=True)
    mu_ = mean_pred.cpu().squeeze()
    sigma_ = sigma_pred.cpu().squeeze() 
    if return_mlp_outputs:
        return mu_, sigma_, z_mu, z_std 
    else:
        return mu_, sigma_   

@torch.no_grad()
def plot_model_accuracy(expt, config, result):
    """ Plot accuracy of model predictions of experimental data

    This provides a qualitative understanding of current model 
    on training data.
    """

    print("Creating plots to visualize training data predictions...")
    iter_plot_dir = config["plot_dir"]+'preds_%d/'%config["iteration"]
    if os.path.exists(iter_plot_dir):
        shutil.rmtree(iter_plot_dir)
    os.makedirs(iter_plot_dir)

    fig, ax = plt.subplots()
    ax.plot(range(len(result["comp_train_loss"])), result["comp_train_loss"])
    ax.plot(range(len(result["comp_eval_loss"])), result["comp_eval_loss"])
    plt.savefig(iter_plot_dir+'mlp_loss.png')
    plt.close() 

    num_samples, c_dim = expt.comps.shape
    for i in range(num_samples):
        fig, axs = plt.subplots(1,3, figsize=(3*4, 4))

        # Plot MLP model predictions of the spectra
        mu, sigma, z_mu, z_std = from_comp_to_spectrum(expt.t, 
                                                        expt.comps[i,:], 
                                                        result["comp_model"], 
                                                        result["np_model"],
                                                        return_mlp_outputs = True
                                                        )
        axs[0].plot(expt.wl, mu)
        minus = (mu-sigma)
        plus = (mu+sigma)
        axs[0].fill_between(expt.wl, minus, plus, color='grey')
        axs[0].scatter(expt.wl, expt.spectra_normalized[i,:], color='k', s=10)
        axs[0].set_title("(MLP) time : %d conc : %.2f"%(expt.comps[i,1], expt.comps[i,0]))
        
        # Plot the Z values comparision between trained and MLP predictions
        labels = []
        def add_label(violin, label):
            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), label))
        mlp_pred = torch.distributions.Normal(z_mu, z_std).sample(torch.Size([100]))
        add_label(axs[1].violinplot(mlp_pred.cpu().numpy(), showmeans=True), label="MLP")

        train_z_samples = torch.distributions.Normal(result["train_z_mean"][i,:], 
                                                     result["train_z_std"][i,:]
                                                    ).sample(torch.Size([100]))
        add_label(axs[1].violinplot(train_z_samples.cpu().numpy(), showmeans=True), label="NP")
        axs[1].legend(*zip(*labels))

        # Plot NP model prediction from trained Z values
        q_train = torch.distributions.Normal(result["train_z_mean"][i,:], 
                                             result["train_z_std"][i,:]
                                            )
        x_target = torch.from_numpy(expt.t).view(1,len(expt.t),1).to(device)
        for j in range(200):
            y_pred_mu, y_pred_sigma = result["np_model"].xz_to_y(x_target, q_train.rsample())
            p_y_pred = torch.distributions.Normal(y_pred_mu, y_pred_sigma)
            mu = p_y_pred.loc.detach()
            axs[2].plot(expt.wl, mu.squeeze().cpu().numpy(), alpha=0.05, c='tab:orange')
        axs[2].scatter(expt.wl, expt.spectra_normalized[i,:], color='k', s=10)
        axs[2].set_title("NP Model")

        plt.savefig(iter_plot_dir+'%d.png'%(i))
        plt.close()

@torch.no_grad()
def plot_iteration(expt, config, result):
    layout = [['A1','A2', 'C', 'C'], 
              ['B1', 'B2', 'C', 'C']
              ]
    bounds = torch.tensor(config["bounds"]).transpose(-1, -2).to(device)
    # plot selected points
    C_train = expt.points
    bounds =  expt.bounds.cpu().numpy()
    C_grid = get_twod_grid(20, bounds)
    fig, axs = plt.subplot_mosaic(layout, figsize=(4*4, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    axs['A1'].scatter(expt.comps[:,0], expt.comps[:,1], marker='x', color='k')
    axs['A1'].scatter(result["comps_new"][:,0], result["comps_new"][:,1], color='tab:green')
    axs['A1'].set_xlabel('C1', fontsize=20)
    axs['A1'].set_ylabel('C2', fontsize=20)    
    axs['A1'].set_title('C sampling')
    axs['A1'].set_xlim([bounds[0,0], bounds[1,0]])
    axs['A1'].set_ylim([bounds[0,1], bounds[1,1]])

    # plot acqf
    C_grid_ = torch.tensor(C_grid).to(device).reshape(len(C_grid),1,2)
    acq_values = result["acqf"](C_grid_).squeeze().cpu().numpy()
    cmap = colormaps["magma"]
    norm = Normalize(vmin=min(acq_values), vmax = max(acq_values))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    axs['A2'].tricontourf(C_grid[:,0], C_grid[:,1], acq_values, cmap=cmap, norm=norm)
    axs['A2'].scatter(result["comps_new"][:,0], result["comps_new"][:,1], marker='x', color='tab:green')    
    divider = make_axes_locatable(axs["A2"])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel('Acqusition value')
    axs['A2'].set_xlabel('C1', fontsize=20)
    axs['A2'].set_ylabel('C2', fontsize=20) 

    
    rids = np.random.choice(C_train.shape[0], 10)
    random_train_comps = C_train[rids,:]
    for ci in random_train_comps:
        mu, _ = from_comp_to_spectrum(expt.t, ci, result["comp_model"], result["np_model"])
        t_ = expt.t
        axs['B2'].plot(t_, mu.cpu().squeeze(), color='grey')
    axs['B2'].set_title('random sample p(y|c)')
    axs['B2'].set_xlabel('t', fontsize=20)
    axs['B2'].set_ylabel('f(t)', fontsize=20) 

    z_samples = -5.0 + 10.0*torch.randn((20, N_LATENT)).to(device)
    for z_sample in z_samples:
        t = torch.from_numpy(t_)
        t = t.view(1, t_.shape[0], 1).to(device)
        mu, _ = result["np_model"].xz_to_y(t, z_sample)
        axs['B1'].plot(t_, mu.cpu().squeeze(), color='grey')
    axs['B1'].set_title('random sample p(y|z)')
    axs['B1'].set_xlabel('t', fontsize=20)
    axs['B1'].set_ylabel('f(t)', fontsize=20) 

    # plot the full evaluation on composition space
    ax = axs['C']
    bounds = expt.bounds.cpu().numpy()
    scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
    scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
    
    ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
    ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    C_grid = get_twod_grid(10, bounds)

    for ci in C_grid:
        mu, sigma = from_comp_to_spectrum(expt.t, ci, result["comp_model"], result["np_model"])
        mu_ = mu.cpu().squeeze().numpy()
        sigma_ = sigma.cpu().squeeze().numpy()
        norm_ci = np.array([scaler_x.transform(ci[0]), scaler_y.transform(ci[1])])
        _inset_spectra(norm_ci, expt.t, mu_, sigma_, ax, show_sigma=True)
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')

    return fig, axs
