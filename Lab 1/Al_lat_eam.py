from my_labutil.src.plugins.lammps import *
from ase.spacegroup import crystal
from ase.build import *
from surface2 import *
import numpy
import matplotlib.pyplot as plt
from ase.io import write


input_template = """
# ---------- 1. Initialize simulation ---------------------
units metal
atom_style atomic
dimension  3
boundary   p p p
read_data $DATAINPUT

# ---------- 2. Specify interatomic potential ---------------------
pair_style eam/alloy
pair_coeff * * $POTENTIAL  Al

#pair_style lj/cut 4.5
#pair_coeff 1 1 0.392 2.620 4.5

# ---------- 3. Run single point calculation  ---------------------
thermo_style custom step pe lx ly lz press pxx pyy pzz
run 0

# -- include optimization of the unit cell parameter
fix 1 all box/relax iso 0.0 vmax 0.001

# -- enable optimization of atomic positions (and the cell)
min_style cg
minimize 1e-10 1e-10 1000 10000

# ---- 4. Define and print useful variables -------------
variable natoms equal "count(all)"
variable totenergy equal "pe"
variable length equal "lx"

print "Total energy (eV) = ${totenergy}"
print "Number of atoms = ${natoms}"
print "Lattice constant (Angstroms) = ${length}"
        """

def make_struc(i, vac, alat, type):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    unitcell = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[alat, alat, alat, 90, 90, 90])
    multiplier = numpy.identity(3) * i
    ase_supercell = make_supercell(unitcell, multiplier)
    #if i == 2:
        #from ase.visualize import view
        #view(ase_supercell)
    if vac == 1:
        ase_supercell.pop(0)
    structure = Struc(ase2struc(ase_supercell))
    return structure


def compute_energy(alat, template, i, vac, type):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    if type == 1:
        potpath = os.path.join(os.environ['LAMMPS_POTENTIALS'], 'Al_zhou.eam.alloy')
        potential = ClassicalPotential(path=potpath, ptype='eam', element=["Al"])
        runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab1", "1C", "1_EAMOpt", str(i), str(alat)))
        struc = make_struc(i=i, vac=vac, alat=alat, type = type)
        output_file = lammps_run(struc=struc, runpath=runpath, potential=potential, intemplate=template, inparam={})
        energy, lattice = get_lammps_energy(outfile=output_file)
        return energy, lattice
    if type == 2:
        potpath = os.path.join(os.environ['LAMMPS_POTENTIALS'], 'Al_zhou.eam.alloy')
        potential = ClassicalPotential(path=potpath, ptype='eam', element=["Al"])
        runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab1", "2C", "2_EAMOpt",str(vac), str(i), str(alat)))
        struc = make_struc(i=i, vac=vac, alat=alat, type = type)
        output_file = lammps_run(struc=struc, runpath=runpath, potential=potential, intemplate=template, inparam={})
        energy, lattice = get_lammps_energy(outfile=output_file)
        return energy, lattice
    if type > 2:
        potpath = os.path.join(os.environ['LAMMPS_POTENTIALS'], 'Al_zhou.eam.alloy')
        potential = ClassicalPotential(path=potpath, ptype='eam', element=["Al"])
        if type == 3:
            runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab1", "3B", "Slab_opt", str(vac), str(i), str(alat)))
            slab = fcc100_2('Al', size=(4, 4, i), vacuum=5, periodic = True)
        if type == 4:
            runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab1", "3B", "Bulk", str(vac), str(i), str(alat)))
            slab = fcc100_2('Al', size=(4, 4, i), periodic = True)
        if i == 4 and type == 3:
            #from ase.visualize import view
            #view(slab)
            write('slab.cif', slab)
        struc = Struc(ase2struc(slab))
        output_file = lammps_run(struc=struc, runpath=runpath, potential=potential, intemplate=template, inparam={})
        energy, lattice = get_lammps_energy(outfile=output_file)
        return energy, lattice

def lattice_scan(i, vac, type):
    if type > 1:
        alat = 4.0857
        energy_list = [compute_energy(i=i, vac=vac, alat=alat, template=input_template, type=type)[0]]
        return min(energy_list)
    else:
        alat_list = numpy.linspace(3.8, 4.3, 50)
        energy_list = [compute_energy(i=i, vac=vac, alat=a, template=input_template, type=type)[0] for a in alat_list]
        lattice = [compute_energy(i=i, vac=vac, alat=a, template=input_template, type=type)[1]for a in alat_list]
        ind = energy_list.index(min(energy_list))
        print("lattice parameter =", alat_list[ind])
        print("minimum energy =", min(energy_list))
        plt.plot(alat_list, energy_list)
        plt.xlabel('Lattice Parameter (Angstroms)')
        plt.ylabel('Energy (eV)')
        plt.title("Energy vs. Lattice Parameter")
        axes = plt.gca()
        axes.set_ylim([-120, -100])
        plt.show()



if __name__ == '__main__':
    type = 2                                                  #if 1, varies energy over linspace with fixed supercell size i
                                                                #if 2, uses fixed lattice parameter with variable supercell size for vacancy formation
    if type == 2:                                              #if 3, constructs slabs with specified dimensions and vacuum and with the same dimension but no vacuum
        min_E_n1 = []
        min_E_perf = []
        E_coh = []
        Ev = []
        for i in range(1,26):
            min_E_perf.append(lattice_scan(i, 0, type))
            E_coh.append(lattice_scan(i,0,type)/(4*(i**3)))

        for i in range(1,26):
            min_E_n1.append(lattice_scan(i, 1, type))

        for i in range(1,26):
           Ev.append(min_E_n1[i-1] - ((4*i**3-1)/(4*i**3))*min_E_perf[i-1])

        print(min_E_perf)
        print(E_coh)
        plt.plot(range(1,26), Ev)
        plt.xlabel('Supercell Multiplication Factor')
        plt.ylabel('Energy (eV)')
        plt.title("Vacancy Formation Energy")
        axes = plt.gca()
        #axes.set_ylim([-5, 0])
        plt.show()

    if type == 1:
        lattice_scan(i=1, vac=0, type=type)

    if type == 3:
        Esurf=[]
        alat = 4.0857
        Etemp=[]
        Etemp2=[]
        r=[]
        for i in range(1,26):
            Etemp.append(lattice_scan(i=i,vac=0,type=type))
            Etemp2.append(lattice_scan(i=i,vac=0,type=4))
            Esurf.append((Etemp[i-1]-Etemp2[i-1])/(2*(2.8637825**2)*(4**2)))
            r.append(2*i)

        plt.plot(range(1,26),Esurf)
        plt.xlabel('Size of Slab')
        plt.ylabel('Energy (eV/A^2)')
        axes = plt.gca()
        axes.set_ylim([0, 0.2])
        plt.title("Surface Energy")
        plt.show()
