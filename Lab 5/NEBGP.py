# initialize gp model
import kernels
import gp
import numpy as np

kernel = kernels.two_plus_three_body
kernel_grad = kernels.two_plus_three_body_grad
hyps = np.array([1, 1, 0.1, 1, 1e-3]) # sig2, ls2, sig3, ls3, noise std
cutoffs = np.array([4.9, 4.9]) # (don't need to optimize for lab)
energy_force_kernel = kernels.two_plus_three_force_en

gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps, cutoffs,
                              energy_force_kernel=energy_force_kernel)

# import calculator and set potential
import os
from ase.calculators.espresso import Espresso
pot_file = os.environ.get('LAMMPS_POTENTIALS') + '/Al_zhou.eam.alloy'
pseudopotentials = {'Al': pot_file}
input_data = {
    'system': {
        'ecutwfc': 29,
        'ecutrho': 143
    },
    'disk_io': 'low'}
calc = Espresso(pseudopotentials = pseudopotentials, tstress = True, tprnfor = True, kpts = (4, 4, 1), input_data = input_data)

#create structure as FCC surface with addsorbate
from ase.build import fcc111, add_adsorbate
slab = fcc111('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Al', 2, 'hcp')
slab.center(vacuum=5.0, axis=2)

#view and set calculator
from ase.visualize import view
view(slab, viewer='x3d')
slab.set_calculator(calc)
print(slab.get_potential_energy())


# make training structure
import struc

training_struc = struc.Structure(cell=slab.cell,
                                 species=['Al']*len(slab),
                                 positions=slab.positions)
training_forces = slab.get_forces()

# add atoms to training database
gp_model.update_db(training_struc, training_forces)
gp_model.set_L_alpha()

# wrap in ASE calculator
from gp_calculator import GPCalculator

gp_calc = GPCalculator(gp_model)

# test on training structure
slab.set_calculator(gp_calc)
GP_forces = slab.get_forces()

# check accuracy by making a parity plot
import matplotlib.pyplot as plt

plt.plot(training_forces.reshape(-1), GP_forces.reshape(-1), '.')
plt.show()