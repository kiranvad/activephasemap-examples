import os, sys, time, shutil, pdb, json
import torch
torch.set_default_dtype(torch.double)
from activephasemap.models.mlp import MLP
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pytorch_tabnet.tab_model import TabNetRegressor

train_x = torch.load("../output/train_x_4.pt", weights_only=True)
train_z_mean = torch.load("../output/train_z_mean_4.pt", weights_only=True)
train_z_std = torch.load("../output/train_z_std_4.pt", weights_only=True)

def plot_predictions(mlp_model, xgb_model, tabnet_model):
    pred_xgb = torch.from_numpy(xgb_model.predict(train_x.numpy()))
    z_mu_xgb, z_std_xgb = pred_xgb[:,:4], pred_xgb[:, 4:]

    z_mu_mlp, z_std_mlp = mlp_model.mlp(torch.tensor(train_x).to(device))

    pred_tabnet = tabnet_model.predict(train_x.to(device))
    z_mu_tabnet, z_std_tabnet = pred_tabnet[:,:4], pred_tabnet[:, 4:]

    plot_dir = "./plots/"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)
    for i in range(train_x.shape[0]):
        fig, ax = plt.subplots()
        labels = []
        def add_label(violin, label):
            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), label))

        mlp_pred = torch.distributions.Normal(z_mu_mlp[i,:], z_std_mlp[i,:]).sample(torch.Size([100]))
        add_label(ax.violinplot(mlp_pred.cpu().numpy(), showmeans=True), label="MLP")

        xgb_pred = torch.distributions.Normal(z_mu_xgb[i,:], z_std_xgb[i,:]).sample(torch.Size([100]))
        add_label(ax.violinplot(xgb_pred.cpu().numpy(), showmeans=True), label="XGB")

        tabnet_pred = torch.distributions.Normal(z_mu_tabnet[i,:], z_std_tabnet[i,:]).sample(torch.Size([100]))
        add_label(ax.violinplot(tabnet_pred.cpu().numpy(), showmeans=True), label="tabnet")

        train_z_samples = torch.distributions.Normal(train_z_mean[i,:],train_z_std[i,:]).sample(torch.Size([100]))
        add_label(ax.violinplot(train_z_samples.cpu().numpy(), showmeans=True), label="Train")
        ax.legend(*zip(*labels))
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
X_xgb = train_x.numpy()
y_xgb = torch.cat((train_z_mean, train_z_std), dim=1).numpy()
X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)

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
    max_epochs=100,
    patience=10,
    batch_size=8,
    virtual_batch_size=4,
    num_workers=0,
    drop_last=False
)

y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Validation RMSE: {rmse:.4f}")

plot_predictions(mlp_model, best_model, tabnet_model)