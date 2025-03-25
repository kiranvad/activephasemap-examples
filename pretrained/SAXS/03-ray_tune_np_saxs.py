import sys, os, pdb, shutil, json, glob
import numpy as np
import tempfile
import torch
from filelock import FileLock
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from math import pi
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
from activephasemap.models.np import NeuralProcess, train_neural_process
from activephasemap.pretrained.helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = os.getcwd() + "/tune/"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def load_dataset(data_loc):
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = SAXSLogLog(root_dir=data_loc)

    return dataset

def train_np(config):
    r_dim = config["r_dim"]
    z_dim = config["z_dim"]
    h_dim = config["h_dim"]
    lr = config["lr"]
    n_blocks = config["n_blocks"]
    batch_size = int(config["batch_size"])

    neuralprocess = NeuralProcess(r_dim, z_dim, h_dim, n_blocks)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            neuralprocess = nn.DataParallel(neuralprocess)
    neuralprocess.to(device)

    PLOT_DIR = SAVE_DIR+'%d_%d_%d_%.2E_%d_%d/'%(r_dim, z_dim, h_dim, lr, n_blocks, batch_size)
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        print("Created %s directory"%PLOT_DIR)
    else:
        print("Directory %s already exists"%PLOT_DIR)

    with open(PLOT_DIR+'config.json', 'w') as fp:
        json.dump(config, fp)   

    dataset = load_dataset(root_dir='/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/SAXS/')
    x_target = torch.linspace(dataset.xrange[0], 
                            dataset.xrange[1], 
                            dataset.n_domain
                            ).reshape(1, dataset.n_domain, 1).to(device)
    data_loader = DataLoader(dataset, 
                            batch_size=batch_size, shuffle=True,
                            collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
                            )

    # Train neural process model
    neuralprocess.training = True
    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=lr)
    epoch_loss = []
    for epoch in range(501):
        neural_process, optimizer, loss_value = train_neural_process(neuralprocess, data_loader,optimizer)

        if (epoch)%plot_epochs_freq==0:
            torch.save(neural_process.state_dict(), PLOT_DIR+"model.pt")
            with torch.no_grad():
                if isinstance(dataset, SAXSLogLog):
                    fig, ax = plt.subplots()
                else:
                    fig, ax = plt.subplots(1,2, figsize=(4*2, 4))
                plot_samples(ax, dataset, neural_process, x_target, z_dim)
                plt.savefig(PLOT_DIR+'prior_%d.png'%(epoch))
                plt.close()

                plot_posterior_samples(x_target, dataset, neuralprocess)
                plt.savefig(PLOT_DIR+'posterior_%d.png'%(epoch))
                plt.close()

        print("Epoch: %d, Loss value : %2.4f"%(epoch, loss_value))
        epoch_loss.append(loss_value)

    torch.save(neuralprocess.state_dict(), PLOT_DIR+'model.pt')
    np.save(PLOT_DIR+'loss.npy', epoch_loss) 
    train.report({"loss": (np_trainer.epoch_loss_history[-1])})

def main(num_samples=10, max_num_epochs=10):
    config = {
        "r_dim": tune.choice([16, 32, 64, 128]),
        "z_dim": tune.choice([4, 8, 16]),
        "h_dim": tune.choice([16, 32, 64, 128]),
        "batch_size": tune.choice([2, 4, 8]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "n_blocks" : tune.choice([2, 4, 6, 8, 10])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_np),
            resources={"cpu": 40, "gpu": 0.5}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
    with open('best_config.json', 'w') as fp:
        json.dump(best_result.config, fp)

main(num_samples=16, max_num_epochs=100)
