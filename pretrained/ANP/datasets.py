import numpy as np
import collections
import gpytorch
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
import glob 

# The (A)NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tensor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context


NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))

class NPRegressionDataset:
    def __init__(self, max_num_context, batch_size, testing=False):
        self.max_num_context = max_num_context 
        self.batch_size = batch_size
        self.testing = testing

    def get_context_targets(self):
        num_context = int(np.random.rand()*(self.max_num_context - 3) + 3)
        num_target = int(np.random.rand()*(self.max_num_context - num_context))

        return num_context, num_target

    def process(self, x_values, y_values):
        num_context, num_target = self.get_context_targets()
        num_total_points = x_values.shape[1]
        idx = torch.randperm(num_total_points)
        if self.testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            context_x = x_values[:, idx[:num_context],:]
            context_y = y_values[:, idx[:num_context]]
        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, idx[:num_target + num_context],:]
            target_y = y_values[:, idx[:num_target + num_context],:]

            # Select the observations
            context_x = x_values[:, idx[:num_context]]
            context_y = y_values[:, idx[:num_context]]

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


class GPCurvesReader(NPRegressionDataset):
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
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing
        super().__init__(max_num_context, batch_size, testing)

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

    def sample(self):
        """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.
    
    Returns:
      A `CNPRegressionDescription` namedtuple.
        """
        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        num_total_points = 400
        x_values = torch.arange(-2, 2, 1.0/100).unsqueeze(0).repeat(self.batch_size, 1)
        x_values = x_values.unsqueeze(-1)
  
        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        if self._random_kernel_parameters:
            l1 = torch.rand((self.batch_size, self._y_size, self._x_size))*(self._l1_scale - 0.1) + 0.1
            sigma_f = torch.rand((self.batch_size, self._y_size))*(self._sigma_scale - 0.1) + 0.1
            # Or use the same fixed parameters for all mini-batches
        else:
            l1 = torch.ones((self.batch_size, self._y_size, self._x_size))*self._l1_scale
            sigma_f = torch.ones((self.batch_size, self._y_size))*self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.linalg.cholesky(kernel)

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(cholesky, torch.randn((self.batch_size, self._y_size, num_total_points, 1)))
        
        # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(3)
        y_values = y_values.permute(0, 2, 1)

        return self.process(x_values, y_values)

class GaussianProcess(NPRegressionDataset):
    def __init__(self, max_num_context=50, batch_size=8, testing=False, num_points = 100):
        self.mean = gpytorch.means.ConstantMean() 
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) 
        self.x = torch.linspace(-2,2, steps=num_points)
        self.num_points = num_points
        super().__init__(max_num_context, batch_size, testing)
        self.shape = (self.batch_size, self.num_points, 1)

    def sample(self):
        mean_x = self.mean(self.x)
        covar_x = self.kernel(self.x)
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        x_values = self.x.repeat(self.batch_size, 1).reshape(*self.shape)
        y_values = dist.rsample(torch.Size([self.batch_size])).reshape(*self.shape)

        return self.process(x_values, y_values) 


class UVVis(NPRegressionDataset):
    def __init__(self, root_dir, max_num_context=50, batch_size=8, testing=False):
        self.dir = root_dir
        self.files = glob.glob(self.dir+'/*.npz')
        super().__init__(max_num_context, batch_size, testing)

    def sample(self):
        rids = np.random.randint(len(self.files), size=self.batch_size)
        x_values, y_values = [], []
        for i in rids:
            npzfile = np.load(self.files[i])
            wl, I = npzfile['wl'], npzfile['I']
            wl = (wl-min(wl))/(max(wl)-min(wl))
            x_values.append(torch.from_numpy(wl)) 
            y_values.append(torch.from_numpy(I))
            
        # shape [batch_size, num_points, 1]
        x_values = torch.stack(x_values, dim=0).unsqueeze(-1).to(device)
        y_values = torch.stack(y_values, dim=0).unsqueeze(-1).to(device)

        return self.process(x_values, y_values) 