import torch 
import matplotlib.pyplot as plt
import gpytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from np import NeuralProcessModel, plot_functions
from datasets import GaussianProcess
import numpy as np 
import os, shutil 

PLOT_DIR = './plots/'
if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.makedirs(PLOT_DIR)
print('Saving the results to %s'%PLOT_DIR)

TRAINING_ITERATIONS = 100000
MAX_CONTEXT_POINTS = 50 
PLOT_AFTER = 10000
REPR_SIZE = 32 
LATENT_SIZE = 8
ATTENTION_TYPE = "multihead"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16

dataset_train = GaussianProcess(batch_size=16, max_num_context=MAX_CONTEXT_POINTS)
dataset_test = GaussianProcess(batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)

# ### Training and Testing
model = NeuralProcessModel(REPR_SIZE, LATENT_SIZE, attn_type=ATTENTION_TYPE).to(device)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_history = []
for itr in range(TRAINING_ITERATIONS):
    model.train()
    data_train = dataset_train.sample()
    (context_x, context_y), target_x = data_train.query
    target_y = data_train.target_y

    optim.zero_grad()
    dist, log_likelihood, kl_loss, loss = model(context_x, context_y, target_x, target_y)
    loss.backward()
    optim.step()
    title = "Iteration %d, loss : %.4f"%(itr, loss.item())
    print(title)
    loss_history.append(loss.item())
    if itr % PLOT_AFTER == 0:
        model.eval()
        with torch.no_grad():
            data_test = dataset_test.sample()
            (context_x, context_y), target_x = data_test.query
            target_y = data_test.target_y
            dist = model(context_x, context_y, target_x)
            fig, ax = plot_functions(target_x.detach().cpu().numpy(),
                           target_y.detach().cpu().numpy(),
                           context_x.detach().cpu().numpy(),
                           context_y.detach().cpu().numpy(),
                           dist.loc.detach().cpu().numpy(),
                           dist.scale.detach().cpu().numpy()
                           )
            ax.set_title(title)
            plt.savefig(PLOT_DIR+'itr_%d.png'%itr)
            plt.close()
np.save(PLOT_DIR+'loss.npy', loss_history)            
n_smooth = 10000
loss_ = np.convolve(loss_history, np.ones(n_smooth)/n_smooth, mode='valid')
fig, ax = plt.subplots()
ax.plot(np.arange(len(loss_)), loss_)
plt.savefig(PLOT_DIR+'loss.png')
plt.close()
