import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from models.attention import MultiHeadAttn, SelfAttn
import numpy as np 

__all__ = ['PoolingEncoder', 'CrossAttnEncoder', 'Decoder']

def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)

class PoolingEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1,
            dim_hid=128, dim_lat=None, self_attn=False,
            pre_depth=4, post_depth=2):
        super().__init__()

        self.use_lat = dim_lat is not None

        self.net_pre = build_mlp(dim_x+dim_y, dim_hid, dim_hid, pre_depth) \
                if not self_attn else \
                nn.Sequential(
                        build_mlp(dim_x+dim_y, dim_hid, dim_hid, pre_depth-2),
                        nn.ReLU(True),
                        SelfAttn(dim_hid, dim_hid))

        self.net_post = build_mlp(dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid,
                post_depth)

    def forward(self, xc, yc, mask=None):
        out = self.net_pre(torch.cat([xc, yc], 2))
        if mask is None:
            out = out.mean(-2)
        else:
            mask = mask.to(xc.device)
            out = (out * mask.unsqueeze(-1)).sum(-2) / \
                    (mask.sum(-1, keepdim=True).detach() + 1e-5)
        if self.use_lat:
            mu, sigma = self.net_post(out).chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.net_post(out)

class CrossAttnEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=None, self_attn=True,
            v_depth=4, qk_depth=2):
        super().__init__()
        self.use_lat = dim_lat is not None

        if not self_attn:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth-2)
            self.self_attn = SelfAttn(dim_hid, dim_hid)

        self.net_qk = build_mlp(dim_x, dim_hid, dim_hid, qk_depth)

        self.attn = MultiHeadAttn(dim_hid, dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid)

    def forward(self, xc, yc, xt, mask=None):
        q, k = self.net_qk(xt), self.net_qk(xc)
        v = self.net_v(torch.cat([xc, yc], -1))

        if hasattr(self, 'self_attn'):
            v = self.self_attn(v, mask=mask)

        out = self.attn(q, k, v, mask=mask)
        if self.use_lat:
            mu, sigma = out.chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return out

class Decoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1,
            dim_enc=128, dim_hid=128, depth=3):
        super().__init__()
        self.fc = nn.Linear(dim_x+dim_enc, dim_hid)
        self.dim_hid = dim_hid

        modules = [nn.ReLU(True)]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, 2*dim_y))
        self.mlp = nn.Sequential(*modules)

    def add_ctx(self, dim_ctx):
        self.dim_ctx = dim_ctx
        self.fc_ctx = nn.Linear(dim_ctx, self.dim_hid, bias=False)

    def forward(self, encoded, x, ctx=None):
        packed = torch.cat([encoded, x], -1)
        hid = self.fc(packed)
        if ctx is not None:
            hid = hid + self.fc_ctx(ctx)
        out = self.mlp(hid)
        mu, sigma = out.chunk(2, -1)
        sigma = 0.1 + 0.9 * F.softplus(sigma)
        return Normal(mu, sigma)

class PositionEmbedder(nn.Module):
    def __init__(self, basis = "fourier", n_freq=32, sigma=1, n_latents = 8):
        super(PositionEmbedder, self).__init__()

        if basis=="bessel":
            self.basis = self.bessel_functions_basis 
            self.n_basis = 3
        else:
            self.basis = self.fourier_basis
            self.n_basis = 2

        self.n_freq = n_freq
        self.sigma = sigma 
        self.n_latents = n_latents

        self.freq = nn.Linear(in_features=1, out_features=self.n_freq)
        with torch.no_grad(): # fix these weights
            wts = torch.normal(mean=0,std=self.sigma, size=(self.n_freq, 1))
            self.freq.weight = nn.Parameter(torch.exp(wts), requires_grad=False)
            self.freq.bias = nn.Parameter(torch.zeros(self.n_freq), requires_grad=False)

        self.layers = nn.Sequential(
            nn.Linear(self.n_basis*self.n_freq, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_latents)
        )

        return
    
    def forward(self, x):
        nb, ns, _ = x.shape
        xn = self.normalize(x)
        xf = self.freq(xn)
        xb = self.basis(xf)
        xe = self.layers(xb)

        return xe

    def normalize(self, x):
        lower, upper = x.min(1, keepdim=True)[0], x.max(1, keepdim=True)[0]

        return (x-lower)/(upper-lower)
    
    def unnormalize(self, x, lower, upper):

        return (upper-lower)*x+lower

    def bessel_functions_basis(self, x):
        eps = 1e-4
        x = self.unnormalize(x, 0.01, 20.0)
        j0 = torch.sin(x)/(x+eps)
        j1 = ( torch.sin(x)/(x**2 + eps) ) - (torch.cos(x)/(x+eps))
        j2 = ( (3.0/x**2) - 1.0 )*torch.sin(x)/(x+eps) - (3*torch.cos(x)/(x**2+eps))
 
        return torch.cat([j0, j1, j2], dim=-1)

    def fourier_basis(self, x):
        x = self.unnormalize(x, -0.5, 0.5)
        return torch.cat([torch.sin(2 * np.pi * x), torch.cos(2 * np.pi * x)], dim=-1)