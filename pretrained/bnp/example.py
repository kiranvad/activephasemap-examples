import torch
import sys, os, pdb, shutil, yaml
import matplotlib.pyplot as plt 
import numpy as np 

from data.gp import *
from data.saxs import SAXS
from utils.log import RunningAverage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from utils.misc import load_module

config = {"model_name" : "anp",
        "sampler_name" : "saxs",
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

PLOT_DIR = './plots/%s-%s'%(config["sampler_name"], config["model_name"])
if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.makedirs(PLOT_DIR)

model_cls = getattr(load_module(f'models/{config["model_name"]}.py'), config["model_name"].upper())
with open(f'configs/{config["model_name"]}.yaml', 'r') as f:
    model_config = yaml.safe_load(f)
model = model_cls(**model_config).to(device)

if config["sampler_name"] =="gp":
    sampler = GPSampler(Matern52Kernel(), 
                        config["sampler_xrange"], 
                        n_domain=config["sampler_n_domain"], 
                        device=device
                        )
elif config["sampler_name"] =="saxs":
    sampler = SAXS(root_dir='/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/SAXS/', device=device) 
else:
    print("Sampler %s is not recognized"%config["sampler_name"])

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
    return x.squeeze().cpu().data.numpy()

def plot(args, sampler, model):
    xp = torch.linspace(sampler.xrange[0], sampler.xrange[1], sampler.n_domain).to(device)
    batch = sampler.sample(
            batch_size=args["plot_batch_size"],
            num_ctx=args["plot_num_ctx"],
            )

    model.eval()
    with torch.no_grad():
        py = model.predict(batch.xc, batch.yc,
                xp[None,:,None].repeat(args["plot_batch_size"], 1, 1),
                num_samples=args["num_samples"])
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

    if mu.dim() == 4: # multi sample
        for i, ax in enumerate(axs):
            for s in range(mu.shape[0]):
                ax.plot(tnp(xp), tnp(mu[s][i]), color='steelblue',
                        alpha=max(0.5/args["num_samples"], 0.1))
                ax.fill_between(tnp(xp), tnp(mu[s][i])-tnp(sigma[s][i]),
                        tnp(mu[s][i])+tnp(sigma[s][i]),
                        color='skyblue',
                        alpha=max(0.2/args["num_samples"], 0.02),
                        linewidth=0.0)
            ax.plot(tnp(batch.x[i]), tnp(batch.y[i]), zorder=mu.shape[0]+2)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                    color='k', label='context', zorder=mu.shape[0]+1)
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                    color='orchid', label='target',
                    zorder=mu.shape[0]+1)
            ax.legend()
    else:
        for i, ax in enumerate(axs):
            ax.plot(tnp(xp), tnp(mu[i]), color='steelblue', alpha=0.5)
            ax.fill_between(tnp(xp), tnp(mu[i]-sigma[i]), tnp(mu[i]+sigma[i]),
                    color='skyblue', alpha=0.2, linewidth=0.0)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                    color='k', label='context')
            ax.plot(tnp(batch.x[i]), tnp(batch.y[i]), 
                    zorder=mu.shape[0]+2)
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                    color='orchid', label='target')
            ax.legend()

    plt.tight_layout()

    return fig, axs

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
        plot(config, sampler, model)
        plt.savefig(PLOT_DIR+"/%d.png"%step)




