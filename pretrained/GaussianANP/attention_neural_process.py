import numpy as np
import matplotlib.pyplot as plt
import collections
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLOT_DIR = './GaussianANP/'
if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.makedirs(PLOT_DIR)
print('Saving the results to %s'%PLOT_DIR)

### Gaussian Process data generator

# The (A)NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tensor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

  Supports vector inputs (x) and vector outputs (y). Kernel is
  mean-squared exponential, using the x-value l2 coordinate distance scaled by
  some factor chosen randomly in a range. Outputs are independent gaussian
  processes.
      """

    def __init__(self,
               batch_size,
               max_num_context,
               x_size=1,
               y_size=1,
               l1_scale=0.6,
               sigma_scale=1.0,
               random_kernel_parameters=True,
               testing=False):
        """Creates a regression dataset of functions sampled from a GP.

    Args:
      batch_size: An integer.
      max_num_context: The max number of observations in the context.
      x_size: Integer >= 1 for length of "x values" vector.
      y_size: Integer >= 1 for length of "y values" vector.
      l1_scale: Float; typical scale for kernel distance function.
      sigma_scale: Float; typical scale for variance.
      random_kernel_parameters: If `True`, the kernel parameters (l1 and sigma) 
          will be sampled uniformly within [0.1, l1_scale] and [0.1, sigma_scale].
      testing: Boolean that indicates whether we are testing. If so there are
          more targets for visualization.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

    Args:
      xdata: Tensor of shape [B, num_total_points, x_size] with
          the values of the x-axis data.
      l1: Tensor of shape [B, y_size, x_size], the scale
          parameter of the Gaussian kernel.
      sigma_f: Tensor of shape [B, y_size], the magnitude
          of the std.
      sigma_noise: Float, std of the noise that we add for stability.

    Returns:
      The kernel, a float tensor of shape
      [B, y_size, num_total_points, num_total_points].
        """
        num_total_points = xdata.shape[1]

    # Expand and take the difference
        xdata1 = xdata.unsqueeze(1)  # [B, 1, num_total_points, x_size]
        xdata2 = xdata.unsqueeze(2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

    # [B, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :])**2

        norm = torch.sum(norm, -1)  # [B, data_size, num_total_points, num_total_points]

    # [B, y_size, num_total_points, num_total_points]
        kernel = ((sigma_f)**2)[:, :, None, None] * torch.exp(-0.5 * norm)

    # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * torch.eye(num_total_points)

        return kernel

    def generate_curves(self):
        """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.
    
    Returns:
      A `CNPRegressionDescription` namedtuple.
        """
        num_context = int(np.random.rand()*(self._max_num_context - 3) + 3)
    # If we are testing we want to have more targets and have them evenly
    # distributed in order to plot the function.
        if self._testing:
            num_target = 400
            num_total_points = num_target
            x_values = torch.arange(-2, 2, 1.0/100).unsqueeze(0).repeat(self._batch_size, 1)
            x_values = x_values.unsqueeze(-1)
    # During training the number of target points and their x-positions are
    # selected at random
        else:
            num_target = int(np.random.rand()*(self._max_num_context - num_context))
            num_total_points = num_context + num_target
            x_values = torch.rand((self._batch_size, num_total_points, self._x_size))*4 - 2
            

    # Set kernel parameters
    # Either choose a set of random parameters for the mini-batch
        if self._random_kernel_parameters:
            l1 = torch.rand((self._batch_size, self._y_size, self._x_size))*(self._l1_scale - 0.1) + 0.1
            sigma_f = torch.rand((self._batch_size, self._y_size))*(self._sigma_scale - 0.1) + 0.1
            
    # Or use the same fixed parameters for all mini-batches
        else:
            l1 = torch.ones((self._batch_size, self._y_size, self._x_size))*self._l1_scale
            sigma_f = torch.ones((self._batch_size, self._y_size))*self._sigma_scale

    # Pass the x_values through the Gaussian kernel
    # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

    # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.linalg.cholesky(kernel)

    # Sample a curve
    # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(cholesky, torch.randn((self._batch_size, self._y_size, num_total_points, 1)))

    # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(3)
        y_values = y_values.permute(0, 2, 1)

        if self._testing:
      # Select the targets
            target_x = x_values
            target_y = y_values

      # Select the observations
            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context]]
            context_y = y_values[:, idx[:num_context]]

        else:
      # Select the targets which will consist of the context points as well as
      # some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

      # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        context_x = context_x.to(device)
        context_y = context_y.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=target_x.shape[1],
        num_context_points=num_context)


# ### Attention module for the decoder
class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_type, n_heads=8):
        super().__init__()
        if attention_type == "uniform":
            self._attention_func = self._uniform_attention
        elif attention_type == "laplace":
            self._attention_func = self._laplace_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W_k = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W_v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W_q = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W = nn.Linear(n_heads*hidden_dim, hidden_dim)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        else:
            raise NotImplementedError
            
    def forward(self, k, v, q):
        rep = self._attention_func(k, v, q)
        return rep
    
    def _uniform_attention(self, k, v, q):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)
        rep = rep.repeat(1, total_points, 1)
        return rep
    
    def _laplace_attention(self, k, v, q, scale=0.5):
        k_ = k.unsqueeze(1)
        v_ = v.unsqueeze(2)
        unnorm_weights = torch.abs((k_ - v_)*scale)
        unnorm_weights = unnorm_weights.sum(dim=-1)
        weights = torch.softmax(unnorm_weights, dim=-1)
        rep = torch.einsum('bik,bkj->bij', weights, v)
        return rep
    
    def _dot_attention(self, k, v, q):
        scale = q.shape[-1]**0.5
        unnorm_weights = torch.einsum('bjk,bik->bij', k, q) / scale
        weights = torch.softmax(unnorm_weights, dim=-1)
        
        rep = torch.einsum('bik,bkj->bij', weights, v)
        return rep
    
    def _multihead_attention(self, k, v, q):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)
            out = self._dot_attention(k_, v_, q_)
            outs.append(out)
        outs = torch.stack(outs, dim=-1)
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep

# ### Encoder models
class DeterministicEncoder(nn.Module):
    
    def __init__(self, 
                 input_dim, 
                 x_dim, 
                 hidden_dim=32, 
                 num_layers=3,
                 attention_type="dot"):
        super(DeterministicEncoder, self).__init__()
        self._input_layer = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        if attention_type is None:
            self.attention =  Attention(hidden_dim, "uniform") 
        else:
            self.attention = Attention(hidden_dim, attention_type)
        self._target_transform = nn.Linear(x_dim, hidden_dim)
        self._context_transform = nn.Linear(x_dim, hidden_dim)
    
    def forward(self, context_x, context_y, target_x):
        d_encoder_input = torch.cat([context_x,context_y], dim=-1)
        ri = self._input_layer(d_encoder_input)
        for layer in self.mlp:
            ri = torch.relu(layer(ri))
        
        x_star = self._target_transform(target_x)
        xi = self._context_transform(context_x)
        r_star = self.attention(xi, ri, x_star)
    
        return r_star
 
class LatentEncoder(nn.Module):
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim=32, 
                 latent_dim=32, 
                 num_layers=3
                 ):
        super(LatentEncoder, self).__init__()
        self._input_layer = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x, y):
        encoder_input = torch.cat([x,y], dim=-1)
        
        si = self._input_layer(encoder_input)
        for layer in self.mlp:
            si = torch.relu(layer(si))
        
        sc = si.mean(dim=1)
    
        mean = self.mean(sc)
        log_sigma = self.log_var(sc)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        return Normal(mean, sigma)

# ### Decoder module
class Decoder(nn.Module):
    
    def __init__(self, 
                 x_dim, 
                 y_dim, 
                 hidden_dim=32, 
                 latent_dim=32, 
                 num_layers=3
                 ):
        super(Decoder, self).__init__()
        self._target_transform = nn.Linear(x_dim, hidden_dim)
        self.mlp = nn.ModuleList([nn.Linear(2*hidden_dim+latent_dim, 2*hidden_dim+latent_dim) for _ in range(num_layers)])
        self.mean = nn.Linear(2*hidden_dim+latent_dim, y_dim)
        self.log_var = nn.Linear(2*hidden_dim+latent_dim, y_dim)
        
    def forward(self, r, z, target_x):
        x = self._target_transform(target_x)
        
        r = torch.cat([r, z, x], dim=-1)
        for layer in self.mlp:
            r = torch.relu(layer(r))
            
        mean = self.mean(r)
        log_sigma = self.log_var(r)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        return Normal(mean, sigma)

# ### Attention-Neural Process Model
class LatentModel(nn.Module):
    
    def __init__(self, 
                 x_dim,
                 y_dim, 
                 hidden_dim=32, 
                 latent_dim=32, 
                 attn_type="dot",
                 n_det_encoder_layers = 3,
                 n_latent_encoder_layers=3,
                 n_decoder_layers=3):
        
        super(LatentModel, self).__init__()
        self.r_encoder = DeterministicEncoder(x_dim + y_dim, 
                                              x_dim, hidden_dim=hidden_dim,
                                              attention_type=attn_type,
                                              num_layers=n_det_encoder_layers
                                              )
                
        self.z_encoder = LatentEncoder(x_dim + y_dim, 
                                     hidden_dim=hidden_dim, 
                                     latent_dim=latent_dim,
                                     num_layers=n_latent_encoder_layers)
        
        self.decoder = Decoder(x_dim, y_dim, 
                               hidden_dim=hidden_dim, 
                               latent_dim=latent_dim,
                               num_layers=n_decoder_layers
                               )
        
    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1)
        
        rc = self.r_encoder(context_x, context_y, target_x)
        q_context = self.z_encoder(context_x, context_y)
        
        if target_y is None:
            z = q_context.rsample()
        else:
            q_target = self.z_encoder(target_x, target_y)
            z = q_target.rsample()
        
        z = z.unsqueeze(1).repeat(1,num_targets,1)
        dist = self.decoder(rc, z, target_x)
        if target_y is not None:
            log_likelihood = dist.log_prob(target_y)
            kl_loss = kl_divergence(q_target, q_context)
            kl_loss = torch.sum(kl_loss, dim=-1, keepdim=True)
            kl_loss = kl_loss.repeat(1, num_targets).unsqueeze(-1)
            loss = -torch.mean((log_likelihood - kl_loss)/num_targets)
            
            return dist, log_likelihood, kl_loss, loss
        else:
            return dist
        


# ### Plotting utilities
def plot_functions(target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points.
  
  Args: 
    target_x: An array of shape [B,num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [B,num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [B,num_contexts,1] that contains 
        the x values of the context points.
    context_y: An array of shape [B,num_contexts,1] that contains 
        the y values of the context points.
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
      """
  # Plot everything
    fig, ax = plt.subplots()
    ax.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    ax.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    ax.plot(context_x[0], context_y[0], 'ko', markersize=10)
    ax.fill_between(
          target_x[0, :, 0],
          pred_y[0, :, 0] - std[0, :, 0],
          pred_y[0, :, 0] + std[0, :, 0],
          alpha=0.2,
          facecolor='#65c9f7',
          interpolate=True)
    
    return fig, ax


# ### Training and Testing
TRAINING_ITERATIONS = 100000
MAX_CONTEXT_POINTS = 50 
PLOT_AFTER = 10000
HIDDEN_SIZE = 128 
LATENT_SIZE = 64
ATTENTION_TYPE = 'multihead' 
LEARNING_RATE = 1e-4
random_kernel_parameters=True 

dataset_train = GPCurvesReader(
    batch_size=16, max_num_context=MAX_CONTEXT_POINTS, random_kernel_parameters=random_kernel_parameters)

dataset_test = GPCurvesReader(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True, random_kernel_parameters=random_kernel_parameters)

model = LatentModel(1, 1, HIDDEN_SIZE, LATENT_SIZE, attn_type=ATTENTION_TYPE).to(device)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_history = []
for itr in range(TRAINING_ITERATIONS):
    model.train()
    data_train = dataset_train.generate_curves()
    (context_x, context_y), target_x = data_train.query
    target_y = data_train.target_y

    optim.zero_grad()
    dist, log_likelihood, kl_loss, loss = model(context_x, context_y, target_x, target_y)
    loss_history.append(loss.item())
    loss.backward()
    optim.step()
    if itr % PLOT_AFTER == 0:
        title = "Iteration %d, loss : %.4f"%(itr, loss.item())
        print(title)
        model.eval()
        with torch.no_grad():
            data_test = dataset_test.generate_curves()
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
            
np.save(PLOT_DIR+'loss_anp.npz', loss_history)


