import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
import numpy as np 
from sklearn import manifold
from scipy.spatial.distance import cdist
seed = np.random.RandomState(seed=2050)
import itertools as it 

class ManifoldGridViz:
    def __init__(self, num_grid_samples, dim):
        self.nx = num_grid_samples 
        self.dim = dim 
        el = np.array([i/self.nx for i in range(self.nx+1)])
        self.grid = np.array([x for x in it.product(el, repeat=self.dim) if np.isclose(np.sum(x),1)])

    def reduce(self, **kwargs):
        self.m = manifold.MDS(n_components=2, **kwargs)
        self.m.fit(self.grid, **kwargs)
        self.pos = self.m.embedding_

        return 

    def look_up(self, x):
        dist = cdist(x.reshape(1,-1), self.grid)
        idx = np.argmin(dist)

        return self.pos[idx,:]