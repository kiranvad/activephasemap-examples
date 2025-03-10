import sys, os, pdb, shutil, json
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from activephasemap.models.np import NeuralProcess, train_neural_process
from activephasemap.pretrained.helpers import *
from typing import Dict
import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

# setup ray cluster
if not "ip_head" in os.environ:
    ray.init()
else:
    ray.init(address='auto', 
             _node_ip_address=os.environ["ip_head"].split(":")[0],
             _redis_password=os.environ["redis_password"]
             )

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/02/"
print("Saving results to: ", PLOT_DIR)
if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.makedirs(PLOT_DIR)

batch_size = 128
r_dim = 128  # Dimension of representation of context points
z_dim = 8  # Dimension of sampled latent variable
h_dim = 128  # Dimension of hidden layers in encoder and decoder
learning_rate = 1e-3

num_epochs = 2
plot_epochs_freq = 5
print_itr_freq = 1000

# Create dataset
dataset = SAXSDataSet(root_dir='/mmfs1/home/kiranvad/cheme-kiranvad/sas-55m-20k')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
)
x, y = next(iter(data_loader))
print('Batch data shape for training : ', x.shape, y.shape)

# Visualize data samples
fig, ax = plt.subplots()
for i in np.random.randint(len(dataset), size=100):
    xi, yi = dataset[i]
    ax.plot(xi.cpu().numpy(), yi.cpu().numpy(), c='b', alpha=0.5)
plt.savefig(PLOT_DIR+'data_samples.png')
plt.close()

def train_func_per_worker(config: Dict):
    resources = ray.get_runtime_context().get_assigned_resources()
    print(resources)
    num_cpus = sum(v for k, v in resources.get("CPU", []))
    num_gpus = sum(v for k, v in resources.get("GPU", []))
    print("Worker is using %d CPUs and %d GPUs"%(num_cpus, num_gpus))

    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    dataloader = config["data_loader"]

    dataloader = ray.train.torch.prepare_data_loader(dataloader)

    model = NeuralProcess(config["r_dim"], config["z_dim"], config["h_dim"])
    model = ray.train.torch.prepare_model(model)
    model.training = True
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Model training loop
    for epoch in range(epochs):
        if ray.train.get_context().get_world_size() > 1:
            # Required for the distributed sampler to shuffle properly across epochs.
            dataloader.sampler.set_epoch(epoch)

        model, optimizer, loss_value = train_neural_process(model, dataloader, optimizer)    
        checkpoint = ray.train.Checkpoint.from_directory(PLOT_DIR)

        ray.train.report(metrics={"loss": loss_value}, checkpoint=checkpoint)

# Configure computation resources
num_workers = 10
total_num_cpus = ray.cluster_resources()['CPU']
total_num_gpus = ray.cluster_resources()['GPU']
print(total_num_cpus, total_num_gpus)
n_cpu_per_worker = (total_num_cpus-4)/num_workers
n_gpu_per_worker = (total_num_gpus)/num_workers
scaling_config = ScalingConfig(num_workers=num_workers,
                               resources_per_worker={ "CPU": n_cpu_per_worker,"GPU": n_gpu_per_worker},
                              use_gpu = torch.cuda.is_available()
                        )

train_config = {
    "lr": learning_rate,
    "epochs": num_epochs,
    "batch_size_per_worker": batch_size // num_workers,
    "data_loader" : data_loader,
    "r_dim" : r_dim,
    "z_dim" : z_dim,
    "h_dim" : h_dim
}

# Initialize a Ray TorchTrainer
trainer = TorchTrainer(
    train_loop_per_worker=train_func_per_worker,
    train_loop_config=train_config,
    scaling_config=scaling_config,
)

# [4] Start distributed training
# Run `train_func_per_worker` on all workers
# =============================================
result = trainer.fit()
print(f"Training result: {result}")

