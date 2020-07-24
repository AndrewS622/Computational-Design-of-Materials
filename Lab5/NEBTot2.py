from ase.calculators.eam import EAM
from ase.calculators.espresso import Espresso
import os
from ase.build import fcc111, add_adsorbate
from ase.visualize import view
from ase.optimize import BFGS
from ase.neb import NEB
import numpy as np
import matplotlib.pyplot as plt
import kernels
import gp
import struc
from gp_calculator import GPCalculator

###EAM Setup, Optimization, and Plotting
pot_file = os.environ.get('LAMMPS_POTENTIALS') + '/Al_zhou.eam.alloy'
pot_file2 = 'Al.pbe-n-rrkjus_psl.1.0.1.UPF'
pseudopotentials = {'Al': pot_file2}
zhou = EAM(potential=pot_file)
slabEAM = fcc111('Al', size=(2, 2, 3))
add_adsorbate(slabEAM, 'Al', 2, 'hcp')
slabEAM.center(vacuum=5.0, axis=2)
slabEAM.set_calculator(zhou)
print(slabEAM.get_calculator())
print(slabEAM.get_potential_energy())
dyn = BFGS(slabEAM)
dyn.run(fmax=0.0001)

slab_2EAM = fcc111('Al', size=(2, 2, 3))
add_adsorbate(slab_2EAM, 'Al', 2, 'fcc')
slab_2EAM.center(vacuum=5.0, axis=2)
slab_2EAM.set_calculator(EAM(potential=pot_file))
dyn = BFGS(slab_2EAM)
slab_2EAM.get_potential_energy()
print(slab_2EAM.get_potential_energy())
dyn.run(fmax=0.0001)

no_images = 7
imagesEAM = [slabEAM]    #first image is the first slab (endpoint)
imagesEAM += [slabEAM.copy() for i in range(no_images-2)]    #copy first slab for intermediate images
imagesEAM += [slab_2EAM]    #final image is the second slab (endpoint 2)
nebEAM = NEB(imagesEAM)
nebEAM.interpolate()
pot_dir = os.environ.get('LAMMPS_POTENTIALS')
for image in imagesEAM[1:no_images-1]:
    image.set_calculator(EAM(potential=pot_file))
optimizer = BFGS(nebEAM)
optimizer.run(fmax=0.01)
pes = np.zeros(no_images)
pos = np.zeros((no_images, len(imagesEAM[0]), 3))
EAMForces = np.zeros((no_images,len(imagesEAM[0]),3))
for n, image in enumerate(imagesEAM):
    pes[n] = image.get_potential_energy()
    pos[n] = image.positions
    EAMForces[n] = image.get_forces()
plt.plot(pes-pes[0], 'k.', markersize=10)  # plot energy difference in eV w.r.t. first image
plt.plot(pes-pes[0], 'k--', markersize=10)
plt.xlabel('image #')
plt.ylabel('energy difference (eV)')
plt.title('EAM')
plt.show()

###DFT Setup, Optimization, and Plotting
pseudopotentials = {'Al': pot_file2}
input_data = {
    'system': {
        'ecutwfc': 29,
        'ecutrho': 143,
        'occupations': 'smearing',
        'smearing': 'mp',
        'degauss': 0.02
    },
    'disk_io': 'low'}
calc = Espresso(pseudopotentials = pseudopotentials, tstress = True, tprnfor = True, kpts = (4, 4, 1), input_data = input_data)
# slabDFT = fcc111('Al', size=(2, 2, 3))
# add_adsorbate(slabDFT, 'Al', 2, 'hcp')
# slabDFT.center(vacuum=5.0, axis=2)
# slabDFT.set_calculator(calc)
# print(slabDFT.get_calculator())
# print(slabDFT.get_potential_energy())
# dyn = BFGS(slabDFT)
# dyn.run(fmax=0.0001)

# slab_2DFT = fcc111('Al', size=(2, 2, 3))
# add_adsorbate(slab_2DFT, 'Al', 2, 'fcc')
# slab_2DFT.center(vacuum=5.0, axis=2)
# calc_2DFT = Espresso(pseudopotentials = pot_file2, tstress = True, tprnfor = True, kpts = (4, 4, 1), input_data = input_data)
# slab_2DFT.set_calculator(calc_2)
# dyn = BFGS(slab_2DFT)
# slab_2DFT.get_potential_energy()
# print(slab_2DFT.get_potential_energy())
# dyn.run(fmax=0.0001)
#
# no_images = 7
# imagesDFT = [slabDFT]    #first image is the first slab (endpoint)
# imagesDFT += [slabDFT.copy() for i in range(no_images-2)]    #copy first slab for intermediate images
# imagesDFT += [slab_2DFT]    #final image is the second slab (endpoint 2)
# nebDFT = NEB(imagesDFT)
# nebDFT.interpolate()
# for image in imagesDFT[1:no_images-1]:
#     image.set_calculator(Espresso(pseudopotentials = pot_file2, tstress = True, tprnfor = True, kpts = (4, 4, 1), input_data = input_data))
# optimizer = BFGS(nebDFT)
# optimizer.run(fmax=0.01)

for image in imagesEAM:
    image.set_calculator(Espresso(pseudopotentials=pseudopotentials, tstress=True, tprnfor=True, kpts=(4, 4, 1), input_data=input_data))

print()
pes = np.zeros(no_images)
pos = np.zeros((no_images, len(imagesEAM[0]), 3))
DFTForces = np.zeros((no_images,len(imagesEAM[0]),3))
for n, image in enumerate(imagesEAM):
    pes[n] = image.get_potential_energy()
    pos[n] = image.positions
    DFTForces[n] = image.get_forces()
plt.plot(pes-pes[0], 'k.', markersize=10)  # plot energy difference in eV w.r.t. first image
plt.plot(pes-pes[0], 'k--', markersize=10)
plt.xlabel('image #')
plt.ylabel('energy difference (eV)')
plt.title('DFT')
plt.show()

plt.plot(DFTForces.reshape(-1),EAMForces.reshape(-1), '.')
plt.xlabel('DFT Force')
plt.ylabel('EAM Force')
plt.show()

###GP Setup, Optimization, and Plotting
# calcGP = Espresso(pseudopotentials = pseudopotentials, tstress = True, tprnfor = True, kpts = (4, 4, 1), input_data = input_data)
# slabGP = fcc111('Al', size=(2, 2, 3))
# add_adsorbate(slabGP, 'Al', 2, 'hcp')
# slabGP.center(vacuum=5.0, axis=2)
# slabGP.set_calculator(calcGP)
# slab_2GP = fcc111('Al', size=(2, 2, 3))
# add_adsorbate(slab_2GP, 'Al', 2, 'fcc')
# slab_2GP.center(vacuum=5.0, axis=2)

kernel = kernels.two_plus_three_body
kernel_grad = kernels.two_plus_three_body_grad
hyps = np.array([1, 1, 0.1, 1, 1e-3]) # sig2, ls2, sig3, ls3, noise std
cutoffs = np.array([4.9, 4.9]) # (don't need to optimize for lab)
energy_force_kernel = kernels.two_plus_three_force_en
gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps, cutoffs,
                              energy_force_kernel=energy_force_kernel)
# training_struc = struc.Structure(cell=slabGP.cell,
#                                  species=['Al']*len(slabGP),
#                                  positions=slabGP.positions)
# training_forces = slabGP.get_forces()
for image in imagesEAM:
    training_struc = struc.Structure(cell = image.cell,
                                     species=['Al'] * len(image),
                                     positions=image.positions)
    training_forces = image.get_forces()
    gp_model.update_db(training_struc, training_forces)
gp_model.set_L_alpha()

for image in imagesEAM:
    image.set_calculator(GPCalculator(gp_model))


# gp_calc = GPCalculator(gp_model)
# calc_2GP = GPCalculator(gp_model)
# slab_2GP.set_calculator(calc_2GP)
# slabGP.set_calculator(gp_calc)
# GP_forces = slabGP.get_forces()
# plt.plot(training_forces.reshape(-1), GP_forces.reshape(-1), '.')
# plt.show()


# no_images = 7
# imagesGP = [slabGP]    #first image is the first slab (endpoint)
# imagesGP += [slabGP.copy() for i in range(no_images-2)]    #copy first slab for intermediate images
# imagesGP += [slab_2GP]    #final image is the second slab (endpoint 2)
# nebGP = NEB(imagesGP)
# nebGP.interpolate()
pes = np.zeros(no_images)
pos = np.zeros((no_images, len(imagesEAM[0]), 3))
GPForces = np.zeros((no_images,len(imagesEAM[0]),3))
for n, image in enumerate(imagesEAM):
    pes[n] = image.get_potential_energy()
    pos[n] = image.positions
    GPForces[n] = image.get_forces()

plt.plot(pes-pes[0], 'k.', markersize=10)  # plot energy difference in eV w.r.t. first image
plt.plot(pes-pes[0], 'k--', markersize=10)
plt.xlabel('image #')
plt.ylabel('energy difference (eV)')
plt.title('GP')
plt.show()

plt.plot(GPForces.reshape(-1),EAMForces.reshape(-1), '.')
plt.xlabel('GP Force')
plt.ylabel('EAM Force')
plt.show()

plt.plot(DFTForces.reshape(-1),GPForces.reshape(-1), '.')
plt.xlabel('DFT Force')
plt.ylabel('GP Force')
plt.show()