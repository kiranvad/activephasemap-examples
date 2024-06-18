import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from torch.utils.data import DataLoader, Dataset
from activephasemap.models.np import NeuralProcess, train_neural_process
from activephasemap.utils.simulators import UVVisExperiment
import json, sys, pdb, glob
import matplotlib.pyplot as plt

sys.path.append('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/')
from helpers import *

class NPModelDataset(Dataset):
    def __init__(self, time, y):
        self.data = []
        for yi in y:
            xi = torch.from_numpy(time).to(device)
            yi = torch.from_numpy(yi).to(device)
            self.data.append((xi.unsqueeze(1),yi.unsqueeze(1)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def finetune_neural_process(data_loader, model, **kwargs):
    batch_size = kwargs.get('batch_size',  16)
    num_iterations = kwargs.get('num_iterations',  30)
    freeze_params, finetune_params = [], []
    finetune_tags = ["_to_hidden.4.weight", "hidden_to"]
    for name, param in model.named_parameters():
        if "_to_hidden.4.weight" in name:
            torch.nn.init.xavier_uniform_(param)
            print("Finetuning %s..."%name)
            finetune_params.append(param)
        elif "hidden_to" in name:
            if "weight" in name:
                torch.nn.init.xavier_uniform_(param)
                finetune_params.append(param)
                print("Finetuning %s..."%name)
        else:
            freeze_params.append(param)

    model.training = True
    lr = kwargs.get('learning_rate',  1e-3)
    optimizer = torch.optim.Adam([{'params': freeze_params, "lr":lr},
                                  {'params': finetune_params, 'lr': lr*10}],
                                  lr=lr
                                )
    epoch_loss = []
    for epoch in range(num_iterations):
        model, optimizer, loss_value = train_neural_process(model, data_loader,optimizer)

        print("Epoch: %d, Loss value : %2.4f"%(epoch, loss_value))
        epoch_loss.append(loss_value)

    # freeze model training
    model.training = False

    return model, epoch_loss

design_space_bounds = [(0.0, 87.0), (0.0,11.0)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)
EXPT_DATA_DIR = "./data/"
ITERATION = len(glob.glob(EXPT_DATA_DIR+"/spectra_*.npy"))
expt = UVVisExperiment(design_space_bounds, ITERATION, EXPT_DATA_DIR)
expt.generate(use_spline=True)

with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/best_config.json') as f:
    config = json.load(f)

batch_size = config["batch_size"]
r_dim = config["r_dim"]  # Dimension of representation of context points
z_dim = config["z_dim"]  # Dimension of sampled latent variable
h_dim = config["h_dim"]  # Dimension of hidden layers in encoder and decoder
learning_rate = config["lr"]

model = NeuralProcess(r_dim, z_dim, h_dim).to(device)
PRETRAIN_LOC = "/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap-examples/pretrained/UVVis/test_np_new_api/model.pt"
model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device))

x_target = torch.linspace(min(expt.t), max(expt.t), 100).reshape(1,100,1).to(device)
with torch.no_grad():
    fig, ax = plt.subplots()
    plot_samples(ax, model, x_target, z_dim)
    plt.savefig('./plots/finetune_before.png')
    plt.close()

dataset = NPModelDataset(expt.t, expt.spectra_normalized)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model, loss = finetune_neural_process(data_loader,
                                      model, 
                                      batch_size=batch_size, 
                                      num_iterations=500, 
                                      learning_rate=learning_rate
                                      )

with torch.no_grad():
    fig, ax = plt.subplots()
    plot_samples(ax, model, x_target, z_dim)
    plt.savefig('./plots/finetune_after.png')
    plt.close()

    # Plot curve fitting-like samples from posteriors
    plot_posterior_samples(x_target, data_loader, model)
    plt.savefig('./plots/finetuned_posterior.png')
    plt.close()