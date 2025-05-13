import torch
import attridict
import numpy as np 
from scipy import interpolate 
import pdb

class SAXS:
    def __init__(self, root_dir, device="cpu"):
        """
        Arguments:
            root_dir (string): Directory with all the data.
        """
        self.dir = root_dir
        data = np.load(self.dir+"sasmodels.npz")
        self.q = data["x"]
        self.Iq = data["y"]
        self.xrange = [-3, 0]
        self.n_domain = 100
        self.q_grid = np.linspace(self.xrange[0], 
                                  self.xrange[1], 
                                  self.n_domain
                                  )
        self.device = device

    def sample(self,
                batch_size=16,
                num_ctx=None,
               ):

        batch = attridict()
        num_ctx = num_ctx or torch.randint(low=3, high=self.n_domain-3, size=[1]).item()
        num_tar = torch.randint(low=3, high=self.n_domain-num_ctx, size=[1]).item()

        locations = np.random.choice(self.n_domain, size=num_ctx + num_tar, replace=False)

        xb, yb = self.get_batch(batch_size)
        batch.x = torch.cat(xb, dim=1).T.unsqueeze(-1)
        batch.y = torch.cat(yb, dim=1).T.unsqueeze(-1)
        
        batch.xc = batch.x[:,locations[:num_ctx]]
        batch.xt = batch.x[:,locations[num_ctx:]]

        batch.yc = batch.y[:,locations[:num_ctx]]
        batch.yt = batch.y[:,locations[num_ctx:]]

        return batch        
    
    def get_batch(self, batch_size):
        xb, yb = [], []
        for _ in range(batch_size):  
            i = np.random.randint(self.Iq.shape[0])
            Iq = self.Iq[i,:]
            spline = interpolate.splrep(np.log10(self.q), np.log10(Iq), s=0)
            I_grid = interpolate.splev(self.q_grid, spline, der=0)

            domain = torch.tensor(self.q_grid).unsqueeze(1)
            codomain = torch.tensor(I_grid).unsqueeze(1)
            xb.append(domain.to(self.device))
            yb.append(codomain.to(self.device))
        
        return xb, yb

            