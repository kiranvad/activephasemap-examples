from activephasemap.models.xgb import XGBoost
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
import pdb, json
import numpy as np
from botorch.utils.sampling import draw_sobol_samples
import matplotlib.pyplot as plt

TRAINING_ITERATIONS = 1000 # total iterations for each optimization
NUM_RESTARTS = 5 # number of optimization from random restarts
LEARNING_RATE = 0.1

def sphere(x1, x2):
    return (x1**2 + x2**2)

# define range for input
r_min, r_max = -5.0, 5.0
design_space_bounds = [(r_min, r_max), (r_min, r_max)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)
num_samples = 1000
x1 = np.random.uniform(low=r_min, high=r_max, size=(num_samples,))
x2 = np.random.uniform(low=r_min, high=r_max, size=(num_samples,))

y = np.asarray([sphere(x1[i], x2[i]) for i in range(num_samples)])

train_x = torch.from_numpy(np.stack((x1, x2))).T.to(device)
train_y = torch.from_numpy(y).view(num_samples, 1).to(device)
print("Input data shapes : ", train_x.shape, train_y.shape)

xgb_model_args = {"objective": "reg:squarederror",
                  "max_depth": 3,
                  "eta": 0.1,
                  "eval_metric": "rmse"
                  }

model = XGBoost(xgb_model_args)
train_loss, eval_loss = model.train(train_x, train_y)

def ground_truth(x):
    x1, x2 = x[...,0], x[...,1]
    euler = 2.718281828459045
    return -20.0 * torch.exp(-0.2 * torch.sqrt(0.5 * (x1**2 + x2**2))) - torch.exp(0.5 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2))) + euler + 20

def mse_loss(y_hat):
    err = (y_hat-0.0)**2

    return err

# Initialize using random Sobol sequence sampling
X = draw_sobol_samples(bounds=bounds, n=NUM_RESTARTS, q=1).to(device)
X.requires_grad_(True)

optimizer = torch.optim.Adam([X], lr=LEARNING_RATE)
X_traj = []

# run a basic optimization loop
for i in range(TRAINING_ITERATIONS):
    optimizer.zero_grad()
    # this performs batch (num_restrats) evaluation
    output = model.predict(X)
    # output = ground_truth(X)

    losses = mse_loss(output[1].squeeze()) 
    loss = losses.sum()

    loss.backward()  
    optimizer.step()
    
    # clamp values to the feasible set
    for j, (lb, ub) in enumerate(zip(*bounds)):
        X.data[..., j].clamp_(lb, ub) 

    # store the optimization trajectory
    # clone and detaching is importat to not meddle with the autograd
    X_traj.append(X.clone().detach())
    if (i + 1) % 100 == 0:
        print(f"Iteration {i+1:>3}/{TRAINING_ITERATIONS:>3} - Loss: {loss.item():>4.3f}; dX: {X.grad.mean()}")

with torch.no_grad():
    X_traj = torch.stack(X_traj, dim=1).squeeze()
    fig, ax = plt.subplots()
    ax.tricontourf(x1, x2, y, cmap="binary")
    for i in range(NUM_RESTARTS):
        traj = X_traj.cpu().numpy()[i,:,:]
        line, = ax.plot(traj[:,0], traj[:,1],lw=2, label="%d"%(i+1))

        ax.scatter(traj[0,0], traj[0,1],
                       s=100,marker='.', color = line.get_color(),
                       zorder=10,lw=2
                       )
        ax.scatter(traj[-1,0], traj[-1,1],
                       s=100,marker='+', color = line.get_color(),
                       zorder=10,lw=2
                       )
    ax.set_xlim(*design_space_bounds[0])
    ax.set_ylim(*design_space_bounds[1])
    ax.legend()
    plt.savefig("xgb_test_trajectories.png")
    plt.close()