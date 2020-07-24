# import calculator and set potential
from ase.calculators.eam import EAM
import os
pot_file = os.environ.get('LAMMPS_POTENTIALS') + '/Al_zhou.eam.alloy'
zhou = EAM(potential=pot_file)

#create structure as FCC surface with addsorbate
from ase.build import fcc111, add_adsorbate
slab = fcc111('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Al', 2, 'hcp')
slab.center(vacuum=5.0, axis=2)

#view and set calculator
from ase.visualize import view
view(slab, viewer='x3d')
slab.set_calculator(zhou)
slab.get_potential_energy()
print(slab.get_calculator())
print(slab.get_potential_energy())

#relax structure
from ase.optimize import BFGS
dyn = BFGS(slab)
dyn.run(fmax=0.0001)

#make second endpoint structure, add adatom and relax
slab_2 = fcc111('Al', size=(2, 2, 3))
add_adsorbate(slab_2, 'Al', 2, 'fcc')
slab_2.center(vacuum=5.0, axis=2)
slab_2.set_calculator(EAM(potential=pot_file))

view(slab_2, viewer='x3d')
dyn = BFGS(slab_2)
slab_2.get_potential_energy()
print(slab_2.get_potential_energy())
dyn.run(fmax=0.0001)

#import NEB
from ase.neb import NEB
import numpy as np

# make band
no_images = 7
images = [slab]    #first image is the first slab (endpoint)
images += [slab.copy() for i in range(no_images-2)]    #copy first slab for intermediate images
images += [slab_2]    #final image is the second slab (endpoint 2)
neb = NEB(images)

# interpolate middle images
neb.interpolate()

# set calculators of middle images
pot_dir = os.environ.get('LAMMPS_POTENTIALS')
for image in images[1:no_images-1]:
    image.set_calculator(EAM(potential=pot_file)) #each image must have its own distinct calculator

# optimize the NEB trajectory
optimizer = BFGS(neb)
optimizer.run(fmax=0.01)

# calculate the potential energy of each image
pes = np.zeros(no_images)
pos = np.zeros((no_images, len(images[0]), 3))
for n, image in enumerate(images):
    pes[n] = image.get_potential_energy()
    pos[n] = image.positions
#reported energy in table below is the energy of the saddle point/middle state
#reported force includes the spring forces as well

import matplotlib.pyplot as plt

plt.plot(pes-pes[0], 'k.', markersize=10)  # plot energy difference in eV w.r.t. first image
plt.plot(pes-pes[0], 'k--', markersize=10)
plt.xlabel('image #')
plt.ylabel('energy difference (eV)')
plt.show()

