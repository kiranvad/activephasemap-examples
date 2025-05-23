import torch
import sys, os, pdb, shutil, yaml
import matplotlib.pyplot as plt 
import numpy as np 

from data.gp import *
from data.saxs import SAXSFFT
from utils.log import RunningAverage
from models.modules import PositionEmbedder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from utils.misc import load_module

config = {"model_name" : "np",
        "sampler_name" : "saxs",
        "pos_n_freq" : 64,
        "pos_sigma" : 1.0,
        "pos_n_latents" : 8,
        "lr" : 5e-4,
        "num_steps" : 10000,
        "batch_size" : 16,
        "num_samples" : 32,
        "sampler_n_domain" : 100,
        "sampler_xrange" : [-2.0, 2.0],
        "print_freq" : 100,
        "eval_freq" : 1000,
        "eval_num_batches": 32,
        "plot_batch_size" : 4,
        "plot_num_ctx":25
}

PLOT_DIR = './plots/fft-%s-%s'%(config["sampler_name"], config["model_name"])
if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.makedirs(PLOT_DIR)

model_cls = getattr(load_module(f'models/{config["model_name"]}.py'), config["model_name"].upper())
with open(f'configs/{config["model_name"]}.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

model_config.update(dim_y = 2)
model = model_cls(None, **model_config).to(device)
print(model_config)

sampler = SAXSFFT(root_dir='../SAXS/', device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["num_steps"])

def eval(args, sampler, model):
    batches = []
    for _ in range(args["eval_num_batches"]):
        batches.append(sampler.sample(batch_size=args["batch_size"]))

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in batches:
            outs = model(batch, num_samples=args["num_samples"])
            for key, val in outs.items():
                ravg.update(key, val)

    print("Evaluation mode : ", ravg.info())

def tnp(x):
    amp_zero = x[0,0] + 1j*x[0,1]
    amp_pos = x[1:,0]+ 1j*x[1:,1] 
    amp_neg = torch.flip(amp_pos, [0])
    full_amps = torch.cat([amp_zero.unsqueeze(0), amp_pos, amp_neg, amp_zero.unsqueeze(0)])
    x_inv = torch.fft.ifft(full_amps).real
    return x_inv.squeeze().cpu().data.numpy()

def plot(args, batch, sampler, model):
    model.eval()
    with torch.no_grad():
        py = model.predict(batch.xc, 
                           batch.yc,
                           batch.x.clone(),
                           num_samples=args["num_samples"]
                        )
        mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

    if args["plot_batch_size"] > 1:
        nrows = max(args["plot_batch_size"]//4, 1)
        ncols = min(4, args["plot_batch_size"])
        fig, axs = plt.subplots(nrows, ncols,
                figsize=(5*ncols, 5*nrows))
        axs = axs.flatten()
    else:
        fig = plt.figure(figsize=(5, 5))
        axs = [plt.gca()]

    for i, ax in enumerate(axs):
        for s in range(mu.shape[0]):
            ax.plot(sampler.q_grid, 
                    tnp(mu[s][i]), 
                    color='steelblue',
                    alpha=max(0.5/args["num_samples"], 0.1)
                )
            ax.fill_between(sampler.q_grid, 
                    tnp(mu[s][i])-tnp(sigma[s][i]),
                    tnp(mu[s][i])+tnp(sigma[s][i]),
                    color='skyblue',
                    alpha=max(0.2/args["num_samples"], 0.02),
                    linewidth=0.0
                )
        pdb.set_trace()
        ax.plot(sampler.q_grid, 
                tnp(batch.y[i]), 
                zorder=mu.shape[0]+2,
                color = "k"
                )

    plt.tight_layout()

    return fig, axs

plot_batch = sampler.sample(
        batch_size=config["plot_batch_size"],
        num_ctx=config["plot_num_ctx"],
        )
ravg = RunningAverage()
print('Total number of parameters: {}\n'.format(
        sum(p.numel() for p in model.parameters()))
        )
for step in range(config["num_steps"]+1):
    model.train()
    optimizer.zero_grad()
    batch = sampler.sample(batch_size=config["batch_size"])
    outs = model(batch, num_samples=config["num_samples"])
    outs.loss.backward()
    optimizer.step()
    scheduler.step()

    for key, val in outs.items():
        ravg.update(key, val)

    if step % config["print_freq"] == 0:
        print("Train mode (%d/%d): "%(step, config["num_steps"]+1), ravg.info())

    if step % config["eval_freq"] == 0:
        eval(config, sampler, model)
        ravg.reset()
        plot(config, plot_batch, sampler, model)
        plt.savefig(PLOT_DIR+"/%d.png"%step)




