import numpy as np 
import matplotlib.pyplot as plt 
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

from botorch.utils.transforms import normalize

from activephasemap.models.np.neural_process import NeuralProcess
from activephasemap.utils.settings import initialize_model
from activephasemap.utils.simulators import UVVisExperiment
from funcshape.functions import Function, SRSF, get_warping_function

import glob

DATA_DIR = "./"
TOTAL_ITERATIONS = len(glob.glob(DATA_DIR+"data/comps_*.npy"))

def load_data_and_models(max_iters):
    design_space_bounds = [(0.0, 75.0),
                       (0.0, 75.0),
                       (0.0, 75.0),
                       (0.0, 75.0), 
                       (0.0, 11.0),
                       ]

    expt = UVVisExperiment(design_space_bounds, max_iters, DATA_DIR+"/data/")
    expt.generate(use_spline=True)
    gp_model_args = {"model":"gp", "num_epochs" : 1, "learning_rate" : 1e-3, "verbose": 1}
    input_dim = expt.dim
    output_dim = 2 

    # Load trained GP model for p(z|c)
    train_x = torch.load(DATA_DIR+'/output/train_x_%d.pt'%max_iters, map_location=device)
    train_y = torch.load(DATA_DIR+'/output/train_y_%d.pt'%max_iters, map_location=device)
    bounds = expt.bounds.to(device)
    normalized_x = normalize(train_x, bounds).to(train_x)
    gp_model = initialize_model(normalized_x, train_y, gp_model_args, input_dim, output_dim, device)
    gp_state_dict = torch.load(DATA_DIR+'/output/gp_model_%d.pt'%(max_iters), map_location=device)
    gp_model.load_state_dict(gp_state_dict)

    # Load trained NP model for p(y|z)
    np_model = NeuralProcess(1, 1, 128, 2, 128).to(device)
    np_model.load_state_dict(torch.load(DATA_DIR+'/output/np_model_%d.pt'%(max_iters), map_location=device))

    return expt, gp_model, np_model


def amplitude_phase_distance(t, f1, f2, **kwargs):
    """Define Amplitude-Phase distance as the loss function. 
    
    """
    t = (t-min(t))/(max(t)-min(t))
    f1 = Function(t, f1.reshape(-1,1))
    f2 = Function(t, f2.reshape(-1,1))

    with torch.no_grad():
        warping, network, error = get_warping_function(f1, f2, **kwargs) 

    q1, q2 = SRSF(f1), SRSF(f2)
    delta = q1.qx-q2.qx
    if delta.sum() == 0:
        amplitude, phase = torch.Tensor([0.0]), torch.Tensor([0.0])
    else:
        network.project()
        gam_dev = network.derivative(t.unsqueeze(-1), h=None)
        q_gamma = q2(t)
        term1 = q1.qx.squeeze()
        term2 = q_gamma.squeeze() * torch.sqrt(gam_dev).squeeze()
        y = (term1 - term2) ** 2

        amplitude = torch.sqrt(torch.trapezoid(y, t))

        theta = torch.trapezoid(torch.sqrt(gam_dev).squeeze(), x=t)
        phase = torch.arccos(torch.clamp(theta, -1, 1))

    return amplitude, phase, [warping, network, error]

def get_accuracy(comps, bounds, time, spectra, gp_model, np_model):
    distances = []
    for i in range(comps.shape[0]):
        comp = comps[i,:].reshape(-1,1)
        c = torch.from_numpy(comp).to(device)

        tt = torch.from_numpy(time).to(device)
        x_target = tt.repeat(comp.shape[0]).view(comp.shape[0], len(time), 1)
        y_true = torch.from_numpy(spectra[i,:]).to(device)

        gp_model.eval()
        normalized_x = normalize(c, bounds.to(c))
        posterior = gp_model.posterior(normalized_x)

        mu = []
        for _ in range(250):
            mu_i, _ = np_model.xz_to_y(x_target, posterior.rsample().squeeze(0))
            mu.append(mu_i)

        y_pred = torch.cat(mu).mean(dim=0, keepdim=True).squeeze()
        optim_kwargs = {"n_iters":50, 
                        "n_basis":10, 
                        "n_layers":10,
                        "domain_type":"linear",
                        "basis_type":"palais",
                        "n_restarts":50,
                        "lr":1e-1,
                        "n_domain":len(time),
                        "eps":0.1
                        }
        amplitude, phase, _ = amplitude_phase_distance(tt, y_true, y_pred, **optim_kwargs)
        distances.append((amplitude + phase).item())

    return np.asarray(distances).mean()

accuracies = {}
for iters in range(1,TOTAL_ITERATIONS):
    expt, gp_model, np_model = load_data_and_models(iters)
    train_mean_accuracy = get_accuracy(expt.comps, 
                                       expt.bounds, 
                                       expt.t, 
                                       expt.spectra_normalized, 
                                       gp_model, 
                                       np_model
                                       )
    
    next_comps = np.load(DATA_DIR+"/data/comps_%d.npy"%(iters))
    next_spectra = np.load(DATA_DIR+"/data/spectra_%d.npy"%(iters))

    wav = np.load(DATA_DIR+"/data/wav.npy")
    next_time = (wav-min(wav))/(max(wav)-min(wav))

    test_mean_accuracy =  get_accuracy(next_comps, 
                                       expt.bounds, 
                                       next_time, 
                                       next_spectra, 
                                       gp_model, 
                                       np_model
                                       )
    print("Iteration %d : Train error : %2.4f \t Test error : %2.4f"%(iters, train_mean_accuracy, test_mean_accuracy))
    accuracies[iters] = (train_mean_accuracy, test_mean_accuracy)

train = np.asarray([y[0] for _,y in accuracies.items()]).squeeze()
test = np.asarray([y[1] for _,y in accuracies.items()]).squeeze()
fig, ax = plt.subplots()
xticks = np.arange(1, TOTAL_ITERATIONS)
ax.plot(xticks, train,  "-o", label="Train Error")
ax.plot(xticks, test,  "-o", label="Evaluation Error")
ax.legend()
xlabels = ["%d"%x for x in xticks]
ax.set_xticks(xticks, labels=xlabels)
ax.set_xlabel("Experimental Iterations")
ax.set_ylabel("Amplitude-Phase distance")
plt.savefig("./plots/accuracy_plot_%d.png"%TOTAL_ITERATIONS)
plt.close()