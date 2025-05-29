from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib import colormaps 
import numpy as np

def _inset_spectra(c, t, ft, ax, **kwargs):
    loc_ax = ax.transLimits.transform(c)
    ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
    ins_ax.plot(t, 
                ft, 
                color = kwargs.get("color", "tab:blue"),
                alpha = kwargs.get("alpha", 1.0),
                )
    limits = kwargs.get("limits", None)
    if limits is not None:
        ins_ax.set_ylim(*limits)
    ins_ax.axis('off')
    
    return 

class MinMaxScaler:
    def __init__(self, min, max):
        self.min = min 
        self.max = max 
        self.range = max-min

    def transform(self, x):
        return (x-self.min)/self.range
    
    def inverse(self, xt):
        return (self.range*xt)+self.min

def scaled_tickformat(scaler, x, pos):
    return '%.2f'%scaler.inverse(x) 

def plot_phasemap(bounds, ax, c, s, scale_axis=True, **kwargs):
    scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
    scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
    if scale_axis:
        ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
        ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    t = np.linspace(0,1, s.shape[1])
    for i, (ci, si) in enumerate(zip(c, s)):
        norm_ci = np.array([scaler_x.transform(ci[0]), scaler_y.transform(ci[1])])
        _inset_spectra(norm_ci,t, si, ax, **kwargs)

    return 
