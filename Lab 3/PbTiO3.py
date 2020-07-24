from my_labutil.src.plugins.pwscf import *
from ase.io import write
from ase import Atoms
import matplotlib.pyplot as plt

def make_struc(alat, displacement):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    lattice = alat * numpy.identity(3)
    symbols = ['Pb', 'Ti', 'O', 'O', 'O']
    sc_pos = [[0,0,0], [0.5,0.5,0.5 + displacement], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    perov = Atoms(symbols=symbols, scaled_positions=sc_pos, cell=lattice)
    # check how your cell looks like
    write('perov.cif', perov)
    structure = Struc(ase2struc(perov))
    return structure



def compute_energy(alat, nk, ecut, disp):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    pseudopots = {'Pb': PseudoPotential(ptype='uspp', element='Pb', functional='LDA', name='Pb.pz-d-van.UPF'),
                  'Ti': PseudoPotential(ptype='uspp', element='Ti', functional='LDA', name='Ti.pz-sp-van_ak.UPF'),
                  'O': PseudoPotential(ptype='uspp', element='O', functional='LDA', name='O.pz-rrkjus.UPF')}
    struc = make_struc(alat=alat, displacement=disp)
    # fix the Pb and Ti atoms in place during relaxation
    constraint = Constraint(atoms={'0': [0, 0, 0]})
    #constraint = Constraint(atoms={'0': [0,0,0], '1': [0,0,0]})
    kpts = Kpoints(gridsize=[nk, nk, nk], option='automatic', offset=True)
    dirname = 'PbTiO3_a_{}_ecut_{}_nk_{}_disp_{}'.format(alat, ecut, nk, disp)
    runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab3/Problem2/relax2", dirname))
    input_params = PWscf_inparam({
        'CONTROL': {
            'calculation': 'relax',
            'pseudo_dir': os.environ['ESPRESSO_PSEUDO'],
            'outdir': runpath.path,
            'tstress': True,
            'tprnfor': True,
            'disk_io': 'none'
        },
        'SYSTEM': {
            'ecutwfc': ecut,
            'ecutrho': ecut * 8,
             },
        'ELECTRONS': {
            'diagonalization': 'david',
            'mixing_beta': 0.7,
            'conv_thr': 1e-7,
        },
        'IONS': {
            'ion_dynamics': 'bfgs'
        },
        'CELL': {},
        })

    output_file = run_qe_pwscf(runpath=runpath, struc=struc,  pseudopots=pseudopots,
                               params=input_params, kpoints=kpts, constraint=constraint, ncpu=1)
    output = parse_qe_pwscf_output(outfile=output_file)
    return output['energy']


def lattice_scan(nk,ecut,alat,disp):
    output = compute_energy(alat=alat, ecut=ecut, nk=nk,disp=disp)
    print(output)
    return output


if __name__ == '__main__':
    # put here the function that you actually want to run
    Q=4

    if Q == 1:
        nk = 4
        ecut = 30
        alat_list = numpy.linspace(3.8, 4.1, 10)
        print(alat_list)
        energy_list = []
        for alat in alat_list:
            energy_list.append(lattice_scan(alat=alat, ecut=ecut, nk=nk, disp=0)/5)
        print(alat_list)
        print(energy_list)
        plt.plot(alat_list, energy_list)
        plt.xlabel('Lattice Parameter (A)')
        plt.ylabel('Energy (eV/atom)')
        axes = plt.gca()
        plt.title("Lattice Parameter Optimization")
        plt.show()

    elif Q == 2:
        nk = 7
        ecut = 30
        alat = 3.8789473684210525
        energy_list=[]
        for disp in numpy.linspace(0.01,0.1,10):
            energy_list.append(lattice_scan(alat=alat,ecut=ecut,nk=nk, disp=disp)/5)
        plt.plot(numpy.linspace(0.01,0.1,10), energy_list)
        plt.xlabel('Relative Displacement')
        plt.ylabel('Energy (eV/atom)')
        axes = plt.gca()
        plt.title("Ti Atom Displacement")
        plt.show()

    elif Q == 3:
        E = []
        Ediff = []
        alat = 3.8789473684210525
        for nk in range(2, 52, 5):
            E.append(lattice_scan(nk=nk, ecut=30, alat=alat,disp=0) / 5)
        for i in range(1, len(E) + 1):
            Ediff.append(E[i - 1] - E[len(E) - 1])
        plt.plot(range(2, 52, 5), E)
        plt.xlabel('Input Grid Size')
        plt.ylabel('Energy (eV/atom)')
        axes = plt.gca()
        plt.title("Energy")
        plt.show()

        plt.plot(range(2, 52, 5), Ediff)
        plt.xlabel('Input Grid Size')
        plt.ylabel('Energy Convergence (eV/atom)')
        axes = plt.gca()
        plt.title("Energy Convergence")
        plt.show()

    elif Q == 4:
        Energy = lattice_scan(nk=7,ecut=30,alat=3.8789473684210525,disp=0.030000000000000006)/5
        print(Energy)