from my_labutil.src.plugins.pwscf import *
from ase.spacegroup import crystal
from ase.io import write
from ase.build import *
from ase import Atoms
import numpy
import matplotlib.pyplot as plt

def make_struc(alat,atom,clat):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    if atom == 'Cu' or atom == 'Au':
        fcccell = bulk(atom, 'fcc', a=alat)
        write('fcc.cif', fcccell)
        print(fcccell, fcccell.get_atomic_numbers())
        structure = Struc(ase2struc(fcccell))
    elif atom == 'CuAu':
        lattice = alat * numpy.identity(3)
        lattice[2][2] = clat
        symbols = ['Cu','Au']
        sc_pos = [[0,0,0],[0.5,0.5,0.5]]
        bctcell = Atoms(symbols=symbols, scaled_positions=sc_pos, cell=lattice)
        write('bct.cif', bctcell)
        print(bctcell, bctcell.get_atomic_numbers())
        structure = Struc(ase2struc(bctcell))
    # check how your cell looks like
    print(structure.species)
    return structure

def compute_energy(alat, nk, ecut, atom, clat):
    """
    Make an input template and select potential and structure, and the path where to run
    """

    potpath = os.path.join(os.environ['ESPRESSO_PSEUDO'], 'Cu.pz-d-rrkjus.UPF')
    pseudopots = {'Cu': PseudoPotential(path=potpath, ptype='uspp', element='Cu',
                                        functional='LDA', name='Cu.pz-d-rrkjus.UPF'),
                  'Au': PseudoPotential(path=potpath, ptype='uspp', element='Au',
                                        functional='LDA', name='Au.pz-d-rrkjus.UPF')
                  }
    struc = make_struc(alat=alat, atom=atom, clat=clat)
    if clat != alat:
        kpts = Kpoints(gridsize=[nk, nk, int(numpy.floor(nk/2))], option='automatic', offset=False)
    else:
        kpts = Kpoints(gridsize=[nk, nk, nk], option='automatic', offset=False)
    dirname = '{}_a_{}_ecut_{}_nk_{}_'.format(atom, alat, ecut, nk)
    runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab3/Problem3/Ediff", dirname))
    input_params = PWscf_inparam({
        'CONTROL': {
            'calculation': 'scf',
            'pseudo_dir': os.environ['ESPRESSO_PSEUDO'],
            'outdir': runpath.path,
            'tstress': True,
            'tprnfor': True,
            'disk_io': 'none',
        },
        'SYSTEM': {
            'ecutwfc': ecut,
            'ecutrho': ecut * 8,
            'occupations': 'smearing',
            'smearing': 'mp',
            'degauss': 0.02
             },
        'ELECTRONS': {
            'diagonalization': 'david',
            'mixing_beta': 0.2,
            'conv_thr': 1e-7,
        },
        'IONS': {
            'ion_dynamics': 'bfgs',
        },
        'CELL': {
            'cell_dynamics': 'bfgs',
            'cell_factor': 4.0,
        },
        })

    output_file = run_qe_pwscf(runpath=runpath, struc=struc,  pseudopots=pseudopots,
                               params=input_params, kpoints=kpts, ncpu=1, nkpool=1)
    output = parse_qe_pwscf_output(outfile=output_file)
    return output['energy']

def lattice_scan(nk,ecut,alat,atom,clat):
    output = compute_energy(alat=alat, ecut=ecut, nk=nk, atom=atom, clat=clat)
    print(output)
    return output

if __name__ == '__main__':
    # put here the function that you actually want to run
    Q=6

    if Q == 1:
        ECu=[]
        EAu=[]
        for alat in numpy.linspace(3.9, 4.1, 10):
            ECu.append(lattice_scan(nk=15,ecut=40,alat=alat,atom='Cu',clat=alat))
            EAu.append(lattice_scan(nk=15,ecut=40,alat=alat,atom='Au',clat=alat))
        plt.plot(numpy.linspace(3.4,3.6,10), ECu)
        plt.xlabel('Lattice Parameter (A)')
        plt.ylabel('Energy (eV/atom)')
        axes = plt.gca()
        plt.title("Cu Lattice Parameter Optimization")
        plt.show()

        plt.plot(numpy.linspace(3.9, 4.1, 10), EAu)
        plt.xlabel('Lattice Parameter (A)')
        plt.ylabel('Energy (eV/atom)')
        axes = plt.gca()
        plt.title("Au Lattice Parameter Optimization")
        plt.show()

    elif Q == 2:
        ECu = []
        EAu = []
        EdiffCu = []
        EdiffAu = []
        alatcu=3.56
        alatau=4.06
        for nk in range(5, 45, 5):
            ECu.append(lattice_scan(nk=nk, ecut=40, alat=alatcu, atom='Cu', clat=alatcu))
            EAu.append(lattice_scan(nk=nk, ecut=40, alat=alatau, atom='Au', clat=alatau))
        for i in range(1, len(EAu) + 1):
            EdiffCu.append(ECu[i - 1] - ECu[len(ECu) - 1])
            EdiffAu.append(EAu[i - 1] - EAu[len(EAu) - 1])
        plt.plot(range(5, 45, 5), ECu)
        plt.xlabel('Input K-Point Grid Dimension')
        plt.ylabel('Energy (eV/atom)')
        axes = plt.gca()
        plt.title("Cu K-Point Grid Optimization")
        plt.show()

        plt.plot(range(5, 45, 5), EdiffCu)
        plt.xlabel('Input Grid Size')
        plt.ylabel('Energy Convergence (eV/atom)')
        axes = plt.gca()
        plt.title("Cu Energy Convergence")
        plt.show()

        plt.plot(range(5, 45, 5), EAu)
        plt.xlabel('Input K-Point Grid Dimension')
        plt.ylabel('Energy (eV/atom)')
        axes = plt.gca()
        plt.title("Au K-Point Grid Optimization")
        plt.show()

        plt.plot(range(5, 45, 5), EdiffAu)
        plt.xlabel('Input Grid Size')
        plt.ylabel('Energy Convergence (eV/atom)')
        axes = plt.gca()
        plt.title("Au Energy Convergence")
        plt.show()

    elif Q == 3:
        Energy=lattice_scan(nk=4,ecut=40, alat = 3.81,atom='CuAu',clat=3.80)
        print(Energy)

    elif Q == 4:
        ECuAu=[]
        Ediff=[]
        alat=2.671769560
        clat=3.921812470
        for nk in range(5, 45, 5):
            ECuAu.append(lattice_scan(nk=nk, ecut=40, alat=alat, atom='CuAu', clat=clat) / 2)
        for i in range(1, len(ECuAu) + 1):
            Ediff.append(ECuAu[i - 1] - ECuAu[len(ECuAu) - 1])
        plt.plot(range(5,45,5), ECuAu)
        plt.xlabel('K-Point Input Dimension')
        plt.ylabel('Energy (eV/atom)')
        axes = plt.gca()
        plt.title("CuAu Energy")
        plt.show()

        plt.plot(range(5, 45, 5), Ediff)
        plt.xlabel('K-Point Input Dimension')
        plt.ylabel('Energy Convergence (eV/atom)')
        axes = plt.gca()
        plt.title("CuAu Energy Convergence")
        plt.show()

    elif Q == 5:
        #BCT Supercell
        alat=4
        clat=8
        lattice = alat * numpy.identity(3)
        lattice[2][2] = clat
        symbols = ['Cu', 'Au']
        sc_pos = [[0, 0, 0], [0.5, 0.5, 0.5]]
        multiplier = 2*numpy.identity(3)
        multiplier[2][2]=1
        pfctcell = Atoms(symbols=symbols, scaled_positions=sc_pos, cell=lattice)
        pfctcell = make_supercell(pfctcell,multiplier)
        write('pfct.cif', pfctcell)

        #PFCT Cell
        alat2=(2*(alat)**2)**(1/2)
        lattice = alat2 * numpy.identity(3)
        lattice[2][2] = clat
        symbols = ['Cu', 'Cu', 'Au', 'Au']
        sc_pos = [[0, 0, 0], [0.5, 0.5, 0],[0,0.5,0.5],[0.5,0,0.5]]
        pfctcell = Atoms(symbols=symbols, scaled_positions=sc_pos, cell=lattice)
        write('pfct2.cif', pfctcell)

        #Regular BCT Cell
        lattice_scan(4,ecut=40,alat=4,atom='CuAu',clat=8)

    elif Q ==6:
        alatCuAu = 2.671769560
        clatCuAu = 3.921812470
        alatCu = 3.56
        alatAu = 4.06
        nk=15
        ECuAu = lattice_scan(nk=nk,ecut=40,alat=alatCuAu,atom='CuAu',clat=clatCuAu)
        ECu = lattice_scan(nk=nk,ecut=40,alat=alatCu,atom='Cu',clat=alatCu)
        EAu = lattice_scan(nk=nk,ecut=40,alat=alatAu,atom='Au',clat=alatAu)
        Ediff = 0.5*(ECuAu-ECu-EAu)
        print(ECuAu)
        print(ECu)
        print(EAu)
        print(Ediff)