from autophasemap import BaseDataSet
import numpy as np
import torch

class AutoPhaseMapDataSet(BaseDataSet):
    def __init__(self, C, q, Iq):
        super().__init__(n_domain=q.shape[0])
        self.t = np.linspace(0,1, num=self.n_domain)
        self.q = q
        self.N = C.shape[0]
        self.Iq = Iq
        self.C = C 

        assert self.N==self.Iq.shape[0], "C and Iq should have same number of rows"
        assert self.n_domain==self.Iq.shape[1], "Length of q should match with columns size of Iq"
    
    def generate(self, process=None):
        if process=="normalize":
            self.F = [self.Iq[i,:]/self.l2norm(self.q, self.Iq[i,:]) for i in range(self.N)]
        elif process=="smoothen":
            self.F = [self._smoothen(self.Iq[i,:]/self.l2norm(self.q, self.Iq[i,:]), window_length=7, polyorder=3) for i in range(self.N)]
        elif process is None:
            self.F = [self.Iq[i,:] for i in range(self.N)]

        assert len(self.F)==self.N, "Total number of functions should match the self.N"    

        return