from pyGDM2 import structures
from pyGDM2 import materials
from pyGDM2 import fields
from pyGDM2 import propagators
from pyGDM2 import core
from pyGDM2 import linear
from pyGDM2 import tools
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep

# create a triangular geometry
step = 6.0
geometry = structures.polygon(step, {"N":3, "S":20}, H=1.0, mesh='hex')

# step = 10
# geometry = structures.sphere(step, R=5, mesh='hex') # this would mean radius = 50 nm (R*step)

material = materials.gold()
struct = structures.struct(step, geometry, material)

field_generator = fields.plane_wave
wavelengths = np.linspace(400, 900, 51)
field_kwargs = dict(theta=[0], inc_angle=180)

efield = fields.efield(field_generator, wavelengths=wavelengths, kwargs=field_kwargs)
n1 = n2 = 1.0
dyads = propagators.DyadsQuasistatic123(n1=n1, n2=n2)
sim = core.simulation(struct, efield, dyads)

## --- run the simulation
sim.scatter()

field_kwargs = tools.get_possible_field_params_spectra(sim)
for i, conf in enumerate(field_kwargs):
    print("config", i, ":", conf)

wl, spectrum = tools.calculate_spectrum(sim, field_kwargs[0], linear.extinct)
area_geom = tools.get_geometric_cross_section(sim)

fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
fig.subplots_adjust(wspace=0.25)

axs[0].scatter(geometry[:,0], geometry[:,1])
axs[0].set_xlabel("x (nm)")
axs[0].set_ylabel("y (nm)")

y = spectrum.T[0]/area_geom
spline = splrep(wl, y)
xt = np.linspace(400, 900, 100) # upsample to 100 points
yt = splev(xt, spline)
ax.scatter(wl, y, s=15, color='k')
ax.plot(xt, yt)
axs[1].set_xlabel("wavelength (nm)")
axs[1].set_ylabel("cross section")
plt.savefig("./plots/pygdm_target.png")
np.savez("target.npz", xt=xt, yt=yt)




