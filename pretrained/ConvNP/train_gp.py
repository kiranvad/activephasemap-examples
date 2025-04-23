import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Union
from IPython.display import clear_output
from PIL import Image
import io 
import warnings
warnings.filterwarnings('ignore')
import pdb 

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from torch.utils import data as tdata
import torch.optim as optim
from torch.distributions.kl import kl_divergence

from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.kernels import RBFKernel, ScaleKernel

torch.set_default_dtype(torch.float64)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PowerFunction(nn.Module):
    def __init__(self, K=1):
        super().__init__()
        self.K = K

    def forward(self, x):
        return torch.cat(list(map(x.pow, range(self.K + 1))), -1)


class Encoder(nn.Module):
    def __init__(self, t, dz):
        super().__init__()

        self.t = t
        self.dz = dz

        self.psi = ScaleKernel(RBFKernel()).to(device)
        self.phi = PowerFunction(K=1).to(device)
        self.pos = nn.Softplus().to(device)

        self.cnn = nn.Sequential(
            nn.Conv1d(3, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(32, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 2*self.dz, 5, 1, 2)
        )

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)

    def forward(self, xc: torch.Tensor, yc: torch.Tensor):
        nb, nx, dx = xc.shape
        h = self.psi(self.t, xc).matmul(self.phi(yc))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)
        h = torch.cat([h0, h1], -1)

        rep = torch.cat([self.t.repeat(xc.size(0), 1, 1).to(xc.device), h], -1).transpose(-1, -2)
        z = self.cnn(rep).mean(dim=-1).view(nb, self.dz, 2)
                
        mu = z[...,0] 
        sigma = 0.1 + 0.9 * torch.sigmoid(z[...,1])

        return MultivariateNormal(mu, scale_tril=sigma.diag_embed())

class Decoder(nn.Module):
    def __init__(self, t, dz):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(dz, 4, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(4, 8, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(8, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 8, 5, 1, 2),
            nn.ReLU(),            
            nn.Conv1d(8, 2, 5, 1, 2)
        )
        self.t = t

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)

        self.pos = nn.Softplus().to(device)
        self.psi_rho = ScaleKernel(RBFKernel()).to(device)

    def forward(self, zl: torch.Tensor, xt: torch.Tensor):
        nz, nb, dz = zl.shape 
        nt = self.t.shape[1]
        m = xt.shape[1]

        zl_t = zl.unsqueeze(0).repeat(nt, 1, 1, 1).view(nb*nz, dz, nt)
        f = self.cnn(zl_t).view(nb, nz, nt, 2).mean(dim=1)

        f_mu, f_sigma = f[...,0], f[...,1]
        psi_d = self.psi_rho(xt, self.t).evaluate()
        mu = torch.einsum('bT,bmT->bm', f_mu, psi_d).view(nb, m, 1)
        sigma = torch.einsum('bT,bmT->bm', self.pos(f_sigma), psi_d).view(nb, m, 1)

        return MultivariateNormal(mu, scale_tril=sigma.diag_embed())    


class ConvLNP1D(nn.Module):
    def __init__(self, lower, upper, z_dim = 4, density=8, nz=64):
        super().__init__()
        self.nz = nz
        t = torch.linspace(start=lower, end=upper, steps=density).reshape(1, -1, 1).to(device)
        
        self.encoder = Encoder(t, z_dim)
        self.decoder = Decoder(t, z_dim)

    def forward(self, 
                xc: torch.Tensor, 
                yc: torch.Tensor, 
                xt: torch.Tensor,
                yt: Union[torch.Tensor, None] = None
                ):
        if self.training:
            qc = self.encoder(xc, yc)
            zl = qc.rsample(torch.Size([self.nz]))
            py = self.decoder(zl, xt)
            qt = self.encoder(xt, yt)

            return py, qc, qt 
        else:
            qc = self.encoder(xc, yc)
            zl = qc.rsample(torch.Size([self.nz]))
            py = self.decoder(zl, xt)

            return py 
        

def train(model, dataloader, optimizer):
    model.train()
    avg_loss = 0

    for xc, yc, xt, yt in dataloader:
        xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)

        optimizer.zero_grad()

        py, qc, qt = model(xc, yc, xt, yt)
        log_likelihood = py.log_prob(yt).mean(dim=0).sum()
        kl = kl_divergence(qt, qc).mean(dim=0).sum()

        loss = -log_likelihood + 0.01*kl

        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

    return avg_loss / len(dataloader.dataset)

class Synthetic1D(tdata.Dataset):
    def __init__(self,
                 length_scale=1.0,
                 output_scale=1.0,
                 num_total_max=50,
                 random_params=False,
                 train=True,
                 data_range=(-2, 2),
                 ):

        self.x_dim = 1
        self.y_dim = 1

        self.length_scale = length_scale
        self.output_scale = output_scale

        self.num_total_max = num_total_max

        self.random_params = random_params
        self.train = train

        self.data_range = data_range

        self.length = 256 if self.train else 1

    def kernel(self, x, length_scale, output_scale, jitter=1e-8):
        r"""
        Args:
            x (Tensor): [num_points, x_dim]
            length_scale (Tensor): [y_dim, x_dim]
            output_scale (Tensor): [y_dim]
            jitter (int): for stability
        """
        num_points = x.size(0)

        x1 = x.unsqueeze(0)  # [1, num_points, x_dim]
        x2 = x.unsqueeze(1)  # [num_points, 1, x_dim]

        diff = x1 - x2
        distance = (diff[None, :, :, :] / length_scale[:, None, None, :]).pow(2).sum(-1).clamp_(min=1e-30).sqrt_()  # [y_dim, num_points, num_points]

        exp_component = torch.exp(-math.sqrt(2.5 * 2) * distance)

        constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        covariance = constant_component * exp_component
        scaled_covariance = output_scale.pow(2)[:, None, None] * covariance  # [y_dim, num_points, num_points]
        scaled_covariance += jitter * torch.eye(num_points)
        
        return scaled_covariance

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        num_context = torch.randint(3, self.num_total_max, size=())
        num_target = torch.randint(3, self.num_total_max, size=())
        return self.sample(num_context, num_target)

    def set_length(self, batch_size):
        self.length *= batch_size

    def sample(self, num_context, num_target):
        r"""
        Args:
            num_context (int): Number of context points at the sample.
            num_target (int): Number of target points at the sample.

        Returns:
            :class:`Tensor`.
                Different between train mode and test mode:

                *`train`: `num_context x x_dim`, `num_context x y_dim`, `num_total x x_dim`, `num_total x y_dim`
                *`test`: `num_context x x_dim`, `num_context x y_dim`, `400 x x_dim`, `400 x y_dim`
        """
        if self.train:
            num_total = num_context + num_target
            x_values = torch.empty(num_total, self.x_dim).uniform_(*self.data_range)
        else:
            lower, upper = self.data_range
            num_total = int((upper - lower) / 0.01 + 1)
            x_values = torch.linspace(self.data_range[0], self.data_range[1], num_total).unsqueeze(-1)

        if self.random_params:
            length_scale = torch.empty(self.y_dim, self.x_dim).uniform_(
                0.1, self.length_scale)  # [y, x]
            output_scale = torch.empty(self.y_dim).uniform_(0.1, self.output_scale)  # [y]
        else:
            length_scale = torch.full((self.y_dim, self.x_dim), self.length_scale)
            output_scale = torch.full((self.y_dim,), self.output_scale)

        # [y_dim, num_total, num_total]
        covariance = self.kernel(x_values, length_scale, output_scale)

        cholesky = psd_safe_cholesky(covariance)

        # [num_total, num_total] x [] = []
        y_values = cholesky.matmul(torch.randn(self.y_dim, num_total, 1)).squeeze(2).transpose(0, 1)

        if self.train:
            context_x = x_values[:num_context, :]
            context_y = y_values[:num_context, :]
        else:
            idx = torch.randperm(num_total)
            context_x = torch.gather(x_values, 0, idx[:num_context].unsqueeze(-1))
            context_y = torch.gather(y_values, 0, idx[:num_context].unsqueeze(-1))

        return context_x, context_y, x_values, y_values
    
class _CustomMapDatasetFetcher(tdata._utils.fetch._BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            num_context = torch.randint(3, self.dataset.num_total_max, size=())
            num_target = torch.randint(3, self.dataset.num_total_max, size=())
            data = [self.dataset.sample(num_context, num_target) for _ in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


tdata._utils.fetch._MapDatasetFetcher.fetch = _CustomMapDatasetFetcher.fetch

def cast_numpy(tensor):
    if not isinstance(tensor, np.ndarray):
        return tensor.numpy()
    return tensor

def plot_model_output(ax, xt, y_pred, std, color, label):
    y_pred = cast_numpy(y_pred).reshape(-1)
    xt = cast_numpy(xt[0]).reshape(-1)
    std = cast_numpy(std).reshape(-1)

    ax.plot(xt, y_pred, color=color, label=label)
    ax.fill_between(xt,y_pred - std,y_pred + std, color=color, alpha=0.2)

    return 


def plot_all(ax, xc, yc, xt, yt, gp_pred, np_pred, support=False):
    gp_y_pred, gp_std = gp_pred
    ax.plot(xc.cpu()[0], yc.cpu()[0], 'o', color='black', label="context")
    ax.plot(xt.cpu()[0], yt.cpu()[0], '-', color='black', label="ground truth")
    plot_model_output(ax, xt.cpu(), gp_y_pred, gp_std, 'green', "GP")
    if support:
        ax.vlines([-2, 2], -3, 3, linestyles='dashed')
    ax.set_ylim(-3, 3)
    plot_model_output(ax, xt.cpu(), np_pred.mean.cpu()[0], np_pred.scale_tril.cpu()[0, :, 0, :], 'purple', "NP")

    return 

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# %%
import sklearn.gaussian_process as gp

def oracle_gp(xc, yc, xt):
    kernel = gp.kernels.ConstantKernel() * gp.kernels.Matern(nu=2.5)
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, alpha=1e-5)
    model.fit(xc[0].cpu().numpy(), yc[0].cpu().numpy())
    y_pred, std = model.predict(xt[0].cpu().numpy(), return_std=True)
    
    return y_pred, std

def validate(model, dataloader):
    model.eval()

    xc, yc, xt, yt = next(iter(dataloader))
    xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)

    gp_pred = oracle_gp(xc, yc, xt)

    with torch.no_grad():
        pred_dist = model(xc, yc, xt)

    rmse = (pred_dist.mean - yt).pow(2).sum(-1).mean()

    fig, ax = plt.subplots()
    plot_all(ax, xc, yc, xt, yt, gp_pred, pred_dist)
    
    return fig, ax, rmse

# %% [markdown]
# ### Training

# %%
batch_size = 8
learning_rate = 1e-3
num_epochs = 200

trainset = Synthetic1D(train=True)
testset = Synthetic1D(train=False, num_total_max=15)
trainset.set_length(batch_size)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

lnp = ConvLNP1D(-2.0, 2.0).to(device)
print("Total parameters %d"%get_n_params(lnp))

optimizer = optim.Adam(lnp.parameters(), lr=learning_rate, weight_decay=1e-5)

frames = []
for epoch in range(num_epochs):
    avg_train_loss = train(lnp, trainloader, optimizer)
    if (epoch+1)%1==0:
        fig, axs, rmse = validate(lnp, testloader)
        output = "(%d) Training Loss : %.2f, RMSE : %.2f"%(epoch, avg_train_loss, rmse)
        print(output)
        fig.suptitle(output)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close()
        clear_output(wait=True)

frames[0].save('convCNP.gif', 
               save_all=True, 
               append_images=frames[1:], 
               duration=200, 
               loop=1)    





