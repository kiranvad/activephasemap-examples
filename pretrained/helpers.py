import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
RNG = np.random.default_rng()
import pdb
import glob

class UVVisDataset(Dataset):
    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the data.
        """
        self.dir = root_dir
        self.files = glob.glob(self.dir+'/*.npz')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            npzfile = np.load(self.files[i])
        except Exception as e:
            print('%s Could not load %s'%(type(e).__name__, self.files[i]))
        wl, I = npzfile['wl'], npzfile['I']
        wl = (wl-min(wl))/(max(wl)-min(wl))
        wl_ = torch.tensor(wl.astype(np.float32)).unsqueeze(1)
        I_ = torch.tensor(I.astype(np.float32)).unsqueeze(1)

        return wl_, I_