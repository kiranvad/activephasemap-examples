import os, sys, time, shutil, pdb, json
import torch
torch.set_default_dtype(torch.double)
from activephasemap.models.mlp import MLP
from activephasemap.models.np import NeuralProcess
from activephasemap.utils.simulators import UVVisExperiment
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pytorch_tabnet.tab_model import TabNetRegressor

ITERATION = 4

train_x = torch.load("../output/train_x_%d.pt"%ITERATION, weights_only=True)
train_z_mean = torch.load("../output/train_z_mean_%d.pt"%ITERATION, weights_only=True)
train_z_std = torch.load("../output/train_z_std_%d.pt"%ITERATION, weights_only=True)

X_xgb = train_x.double().cpu().numpy()
y_xgb = torch.cat((train_z_mean, train_z_std), dim=1).double().cpu().numpy()
X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)

# Specify the Neural Process model
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
np_model = NeuralProcess(best_np_config["r_dim"], 
                         best_np_config["z_dim"], 
                         best_np_config["h_dim"]
                         ).to(device)
np_model.load_state_dict(torch.load("../output/np_model_%d.pt"%ITERATION, 
                                    map_location=device, 
                                    weights_only=True
                                    )
                        )

design_space_bounds = [(0.0, 87.0), (0.0,11.0)]
EXPT_DATA_DIR = "../data/"
expt = UVVisExperiment(design_space_bounds, EXPT_DATA_DIR)
expt.read_iter_data(ITERATION)
expt.generate(use_spline=True)

def from_latents_to_spectrum(z_mu, z_std):
    z_dist = torch.distributions.Normal(z_mu, z_std)
    z = z_dist.sample(torch.Size([100]))
    t = torch.from_numpy(expt.t).repeat(100, 1, 1).to(device)
    t = torch.swapaxes(t, 1, 2)
    y_samples, _ = np_model.xz_to_y(t, z)

    mean_pred = y_samples.mean(dim=0, keepdim=True)
    sigma_pred = y_samples.std(dim=0, keepdim=True)
    mu_ = mean_pred.cpu().squeeze()
    sigma_ = sigma_pred.cpu().squeeze() 

    return mu_.detach().numpy(), sigma_.detach().numpy()

def plot_predictions(mlp_model, xgb_model, tabnet_model):
    pred_xgb = torch.from_numpy(xgb_model.predict(train_x.numpy())).to(device)
    z_mu_xgb, z_std_xgb = pred_xgb[:,:4], pred_xgb[:, 4:]

    z_mu_mlp, z_std_mlp = mlp_model.mlp(train_x.double().to(device))

    pred_tabnet = torch.from_numpy(tabnet_model.predict(X_xgb)).to(device)
    z_mu_tabnet, z_std_tabnet = pred_tabnet[:,:4], torch.abs(pred_tabnet[:, 4:])

    plot_dir = "./plots/"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)
    for i in range(train_x.shape[0]):
        fig, axs = plt.subplots(1,2, figsize=(2*4, 4))
        axs[1].scatter(expt.wl, expt.spectra_normalized[i,:], color='k', s=10)
        labels = []
        def add_label(violin, label):
            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), label))

        mlp_pred = torch.distributions.Normal(z_mu_mlp[i,:], z_std_mlp[i,:]).sample(torch.Size([100]))
        add_label(axs[0].violinplot(mlp_pred.cpu().numpy(), showmeans=True), label="MLP")
        mu, sigma = from_latents_to_spectrum(z_mu_mlp[i,:], z_std_mlp[i,:])
        axs[1].plot(expt.wl, mu, label="MLP")
        axs[1].fill_between(expt.wl, mu-sigma, mu+sigma, color='grey')

        xgb_pred = torch.distributions.Normal(z_mu_xgb[i,:], z_std_xgb[i,:]).sample(torch.Size([100]))
        add_label(axs[0].violinplot(xgb_pred.cpu().numpy(), showmeans=True), label="XGB")
        mu, sigma = from_latents_to_spectrum(z_mu_xgb[i,:], z_std_xgb[i,:])
        axs[1].plot(expt.wl, mu, label="XGB")
        axs[1].fill_between(expt.wl, mu-sigma, mu+sigma, color='grey')

        tabnet_pred = torch.distributions.Normal(z_mu_tabnet[i,:], z_std_tabnet[i,:]).sample(torch.Size([100]))
        add_label(axs[0].violinplot(tabnet_pred.cpu().numpy(), showmeans=True), label="tabnet")
        mu, sigma = from_latents_to_spectrum(z_mu_tabnet[i,:], z_std_tabnet[i,:])
        axs[1].plot(expt.wl, mu, label="tabnet")
        axs[1].fill_between(expt.wl, mu-sigma, mu+sigma, color='grey')

        train_z_samples = torch.distributions.Normal(train_z_mean[i,:],train_z_std[i,:]).sample(torch.Size([100]))
        add_label(axs[0].violinplot(train_z_samples.cpu().numpy(), showmeans=True), label="Train")
        axs[0].legend(*zip(*labels))
        axs[1].legend()
        plt.savefig(plot_dir+'%d.png'%(i))
        plt.close()

# 1. Simple MLP method (current)
mlp_model_args = {"num_epochs" : 1000, 
                 "learning_rate" : 1e-3, 
                 "verbose": 100,
                 }
mlp_model = MLP(train_x, train_z_mean, train_z_std, **mlp_model_args)
comp_train_loss, comp_eval_loss = mlp_model.fit(use_early_stoping=True)

# 2. XGBoost 

xgb_model = xgb.XGBRegressor(tree_method="hist",
                             device="cuda",
                             objective='reg:squarederror',
                             eval_metric='rmse'
                             )
search = GridSearchCV(
    xgb_model,
    {"max_depth": [2, 4, 6], 
    "n_estimators": [50, 100, 200]
    },
    scoring='neg_mean_squared_error',
    verbose=10,
)
evalset = [(X_train, y_train), (X_test, y_test)]
search.fit(X_xgb, y_xgb, eval_set=evalset)

best_model = search.best_estimator_
print(f"Best Hyperparameters: {search.best_params_}")

y_pred = best_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Validation RMSE: {rmse:.4f}")

# 3.TabNet regression
torch.set_default_dtype(torch.float)
tabnet_model = TabNetRegressor(
    n_d=8, n_a=8,  # Dimension of decision and attention steps
    n_steps=3,
    gamma=1.3,
    n_independent=2,
    n_shared=2,
    lambda_sparse=0.001,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    mask_type='sparsemax'
)

tabnet_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['rmse'],
    max_epochs=1000,
    patience=100,
    batch_size=32,
    virtual_batch_size=16,
    num_workers=0,
    drop_last=False
)

y_pred = tabnet_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Validation RMSE: {rmse:.4f}")

plot_predictions(mlp_model, best_model, tabnet_model)