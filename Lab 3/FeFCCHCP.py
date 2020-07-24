from my_labutil.src.plugins.pwscf import *
from ase.spacegroup import crystal
from ase.io import write
from ase.build import bulk
import numpy
import matplotlib.pyplot as plt

def make_struc(alat,clat):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    if alat != clat:
        fecell = bulk('Fe', 'hcp', a=alat, c=clat)
        write('fehcp.cif', fecell)
    else:
        fecell = bulk('Fe', 'bcc', a=alat, cubic=True)
        write('febcc.cif', fecell)
    # check how your cell looks like

    print(fecell, fecell.get_atomic_numbers())
    fecell.set_atomic_numbers([26, 27])
    structure = Struc(ase2struc(fecell))
    print(structure.species)
    return structure


def compute_energy(alat, nk, ecut, clat, SM):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    potname = 'Fe.pbe-nd-rrkjus.UPF'
    potpath = os.path.join(os.environ['ESPRESSO_PSEUDO'], potname)
    pseudopots = {'Fe': PseudoPotential(path=potpath, ptype='uspp', element='Fe',
                                        functional='GGA', name=potname),
                  'Co': PseudoPotential(path=potpath, ptype='uspp', element='Fe',
                                        functional='GGA', name=potname)
                  }
    struc = make_struc(alat=alat,clat=clat)
    if clat != alat:
        kpts = Kpoints(gridsize=[nk, nk, int(numpy.floor(nk/2))], option='automatic', offset=False)
    else:
        kpts = Kpoints(gridsize=[nk, nk, nk], option='automatic', offset=False)
    dirname = 'Fe_c_{}_ecut_{}_nk_{}_SM_{}'.format(clat, ecut, nk, SM)
    runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab3/Problem1/BCC/Mag", dirname))
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
            'ecutrho': ecut * 10,
            'nspin': 2,
            'starting_magnetization(1)': SM,
            'starting_magnetization(2)': SM,
            'occupations': 'smearing',
            'smearing': 'mp',
            'degauss': 0.02
             },
        'ELECTRONS': {
            'diagonalization': 'david',
            'mixing_beta': 0.5,
            'conv_thr': 1e-7,
        },
        'IONS': {},
        'CELL': {},
        })

    output_file = run_qe_pwscf(runpath=runpath, struc=struc,  pseudopots=pseudopots,
                               params=input_params, kpoints=kpts, ncpu=1, nkpool=1)
    output = parse_qe_pwscf_output(outfile=output_file)
    return output['energy']

def compute_energy_anti(alat, nk, ecut, clat):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    potname = 'Fe.pbe-nd-rrkjus.UPF'
    potpath = os.path.join(os.environ['ESPRESSO_PSEUDO'], potname)
    pseudopots = {'Fe': PseudoPotential(path=potpath, ptype='uspp', element='Fe',
                                        functional='GGA', name=potname),
                  'Co': PseudoPotential(path=potpath, ptype='uspp', element='Fe',
                                        functional='GGA', name=potname)
                  }
    struc = make_struc(alat=alat,clat=clat)
    if clat != alat:
        kpts = Kpoints(gridsize=[nk, nk, int(numpy.floor(nk/2))], option='automatic', offset=False)
    else:
        kpts = Kpoints(gridsize=[nk, nk, nk], option='automatic', offset=False)
    dirname = 'Fe_a_{}_ecut_{}_nk_{}'.format(alat, ecut, nk)
    runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab3/Problem1/JUNK", dirname))
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
            'ecutrho': ecut * 10,
            'nspin': 2,
            'starting_magnetization(1)': 1,
            'starting_magnetization(2)': -1,
            'occupations': 'smearing',
            'smearing': 'mp',
            'degauss': 0.02
             },
        'ELECTRONS': {
            'diagonalization': 'david',
            'mixing_beta': 0.5,
            'conv_thr': 1e-7,
        },
        'IONS': {},
        'CELL': {},
        })

    output_file = run_qe_pwscf(runpath=runpath, struc=struc,  pseudopots=pseudopots,
                               params=input_params, kpoints=kpts, ncpu=1, nkpool=1)
    output = parse_qe_pwscf_output(outfile=output_file)
    return output['energy']


def lattice_scan(nk,ecut,alat,clat,anti,SM):
    if anti:
        output = compute_energy_anti(alat=alat, ecut=ecut, nk=nk, clat=clat)
    else:
        output = compute_energy(alat=alat, ecut=ecut, nk=nk, clat=clat,SM=SM)
    print(output)
    return output

if __name__ == '__main__':
    # put here the function that you actually want to run
    P=5
    Q=1
    #Xtal set to 1 for HCP and 2 for BCC
    if Q == 1:
        E=[]
        if P == 1:
            Xtal=2
            for alat in numpy.linspace(2.8,2.9,5):
                if Xtal == 2:
                    clat=alat
                elif Xtal == 1:
                    clat = alat*(8/3)**(1/2)
                E.append(lattice_scan(nk=20,ecut=30,alat=alat, clat=clat)/2)
            plt.plot(numpy.linspace(2.8,2.9,5), E)
            plt.xlabel('Lattice Parameter (A)')
            plt.ylabel('Energy (eV/atom)')
            axes = plt.gca()
            plt.title("Lattice Parameter Optimization")
            plt.show()
        elif P == 2:
            alat=2.47
            for clat in numpy.linspace(4.0,4.2,10):
                E.append(lattice_scan(nk=4,ecut=30,alat=alat,clat=clat)/2)
            plt.plot(numpy.linspace(4.0,4.2,10), E)
            plt.xlabel('Lattice Parameter (A)')
            plt.ylabel('Energy (eV/atom)')
            axes = plt.gca()
            plt.title("Lattice Parameter Optimization")
            plt.show()
        elif P == 3:
            E=[]
            Ediff=[]
            alat = 2.47
            clat = 4.13
            for nk in range(2,52,5):
                E.append(lattice_scan(nk=nk,ecut=30,alat=alat,clat=clat)/2)
            for i in range(1,len(E)+1):
                Ediff.append(E[i-1]-E[len(E)-1])
            plt.plot(range(2,52,5), E)
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
        elif P == 4:
            E=[]
            Ediff=[]
            alat = 2.83
            clat=alat
            for nk in range(2,52,5):
                E.append(lattice_scan(nk=nk,ecut=30,alat=alat,clat=clat)/2)
            for i in range(1,len(E)+1):
                Ediff.append(E[i-1]-E[len(E)-1])
            plt.plot(range(2,52,5), E)
            plt.xlabel('Input Grid Size')
            plt.ylabel('Energy (eV)')
            axes = plt.gca()
            plt.title("Energy")
            plt.show()

            plt.plot(range(2, 52, 5), Ediff)
            plt.xlabel('Input Grid Size')
            plt.ylabel('Energy Convergence')
            axes = plt.gca()
            plt.title("Energy Convergence")
            plt.show()
        elif P == 5:
            E=[]
            E2=[]
            alatbcc=2.83
            alathcp=2.47
            Vbcc = alatbcc**3
            ac=1.67
            Vhcp = ac*0.5*3**(1/2)*alathcp**3
            for dV in numpy.linspace(1,11,10):
                abcc = (Vbcc-dV)**(1/3)
                ahcp = (2*(Vhcp-dV)/(ac*3**(1/2)))**(1/3)
                chcp=ac*ahcp
                E.append(lattice_scan(nk=20,ecut=30,alat=abcc,clat=abcc,anti=True,SM=0.7)/2)
                E2.append(lattice_scan(nk=20,ecut=30,alat=ahcp,clat=chcp,anti=True,SM=0.0)/2)

            plt.plot(alatbcc**3-numpy.linspace(1,11,10), E, 'r')
            plt.plot(ac * 0.5 * 3 ** (1 / 2) * alathcp ** 3 - numpy.linspace(1,11,10), E2, 'b')
            plt.xlabel('Unit Cell Volume')
            plt.ylabel('Energy (eV/atom)')
            axes = plt.gca()
            plt.title("Energy")
            plt.show()
        elif P == 6:
            nk=20
            Eanti=[]
            Enon=[]
            Eferro=[]
            for alat in numpy.linspace(2.5,3.0,10):
                Eanti.append(lattice_scan(nk=nk,ecut=30,alat=alat,clat=alat,anti=True,SM=0)/2)
                Enon.append(lattice_scan(nk=nk, ecut=30, alat=alat, clat=alat, anti=False,SM=0.0)/2)
                Eferro.append(lattice_scan(nk=nk,ecut=30,alat=alat,clat=alat,anti=False,SM=0.7)/2)
            plt.plot(numpy.linspace(2.5,3.0,10),Eanti,'r')
            plt.plot(numpy.linspace(2.5,3.0,10),Enon,'b')
            plt.plot(numpy.linspace(2.5,3.0,10),Eferro,'g')
            plt.xlabel('Lattice Parameter (A)')
            plt.ylabel('Energy (eV/atom)')
            plt.title("Ground State Energy")
            plt.show()