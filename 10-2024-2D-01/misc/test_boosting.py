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
from torchensemble.gradient_boosting import GradientBoostingRegressor 
from torchensemble.utils.logging import set_logger

ITERATION = 5
EXPT_DIR = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/test-aunp-sim"
train_x = torch.load(EXPT_DIR+"/output/train_x_%d.pt"%ITERATION, weights_only=True)
train_z_mean = torch.load(EXPT_DIR+"/output/train_z_mean_%d.pt"%ITERATION, weights_only=True)
train_z_std = torch.load(EXPT_DIR+"/output/train_z_std_%d.pt"%ITERATION, weights_only=True)

X_xgb = train_x
y_xgb = torch.cat((train_z_mean, train_z_std), dim=1)
X_train, X_test, y_train, y_test = train_test_split(X_xgb.double().cpu().numpy(), 
                                                    y_xgb.double().cpu().numpy(), test_size=0.2, random_state=42)

# Specify the Neural Process model
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    best_np_config = json.load(f)
np_model = NeuralProcess(best_np_config["r_dim"], 
                         best_np_config["z_dim"], 
                         best_np_config["h_dim"]
                         ).to(device)
np_model.load_state_dict(torch.load(EXPT_DIR+"/output/np_model_%d.pt"%ITERATION, 
                                    map_location=device, 
                                    weights_only=True
                                    )
                        )

design_space_bounds = [(0.0, 87.0), (0.0,11.0)]
expt = UVVisExperiment(design_space_bounds, EXPT_DIR+"/data/")
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

def plot_predictions(mlp_model, xgb_model, tabnet_model, ensemble_model):
    pred_xgb = torch.from_numpy(xgb_model.predict(train_x.numpy())).to(device)
    z_mu_xgb, z_std_xgb = pred_xgb[:,:4], pred_xgb[:, 4:]

    z_mu_mlp, z_std_mlp = mlp_model.mlp(train_x.double().to(device))

    pred_ensemble = ensemble_model.predict(train_x.double().to(device)).to(device)
    z_mu_ensemble, z_std_ensemble = pred_ensemble[:,:4], pred_ensemble[:, 4:]

    torch.set_default_dtype(torch.float)
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

        ensemble_pred = torch.distributions.Normal(z_mu_ensemble[i,:], z_std_ensemble[i,:]).sample(torch.Size([100]))
        add_label(axs[0].violinplot(ensemble_pred.cpu().numpy(), showmeans=True), label="tabnet")
        mu, sigma = from_latents_to_spectrum(z_mu_ensemble[i,:], z_std_ensemble[i,:])
        axs[1].plot(expt.wl, mu, label="ensemble")
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
                             objective='reg:squarederror',
                             eval_metric='rmse'
                             )
search = GridSearchCV(
    xgb_model,
    {"max_depth": [2, 4, 6], 
    "n_estimators": [50, 100, 200]
    },
    scoring='neg_mean_squared_error',
    verbose=False,
)
evalset = [(X_train, y_train), (X_test, y_test)]
search.fit(X_xgb, y_xgb)

best_model = search.best_estimator_
print(f"Best Hyperparameters: {search.best_params_}")

y_pred = best_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"(XGBoost) Validation RMSE: {rmse:.4f}")

# 3.TabNet regression
torch.set_default_dtype(torch.float)
tabnet_model = TabNetRegressor(n_d=64, 
                               n_a=64, 
                               n_steps=10,
                               n_independent = 3,
                               n_shared = 3,
                               verbose=0
                               )

tabnet_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['rmse'],
    max_epochs=1000,
    patience=100,
    batch_size=16,
    virtual_batch_size=8,
    num_workers=0,
    drop_last=False
)

y_pred = tabnet_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"(TabNet) Validation RMSE: {rmse:.4f}")

# 4. Ensemble PyTorch
torch.set_default_dtype(torch.double)
# logger = set_logger('pytorch_ensemble')

class MLPModel(torch.nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layer1 = torch.nn.Linear(x_dim, 32)
        self.layer2 = torch.nn.Linear(32, 32)
        self.layer3 = torch.nn.Linear(32, 16)
  
        self.hidden_to_mu = torch.nn.Linear(16, z_dim)
        self.hidden_to_std = torch.nn.Linear(16, z_dim)

    def forward(self, x):
        h = torch.nn.functional.relu(self.layer1(x))
        h = torch.nn.functional.relu(self.layer2(h))
        h = torch.nn.functional.relu(self.layer3(h))
        mu = self.hidden_to_mu(h)
        std = torch.nn.functional.softplus(self.hidden_to_std(h))
        
        return torch.cat((mu, std), dim=1)

MLP = MLPModel(train_x.shape[-1], best_np_config["z_dim"])
ensemble_model = GradientBoostingRegressor(
    estimator=MLP,
    n_estimators=50,
    cuda=True,
)

criterion = torch.nn.MSELoss(reduction="mean")
ensemble_model.set_criterion(criterion)

ensemble_model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4) 
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).to(device), 
                                               torch.from_numpy(y_train).to(device)
                                               )
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


ensemble_model.fit(train_loader=train_loader,  epochs=100)

y_pred = ensemble_model.predict(torch.from_numpy(X_test).to(device)).numpy()
rmse = root_mean_squared_error(y_test, y_pred)
print(f"(Ensemble) Validation RMSE: {rmse:.4f}")
 
plot_predictions(mlp_model, best_model, tabnet_model, ensemble_model)
