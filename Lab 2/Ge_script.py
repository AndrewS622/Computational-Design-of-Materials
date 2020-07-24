from my_labutil.src.plugins.pwscf import *
from ase.spacegroup import crystal
from ase.build import *
from ase.io import write
import matplotlib.pyplot as plt
import numpy as np


def make_struc(alat, a):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    # set primitive_cell=False if you want to create a simple cubic unit cell with 8 atoms
    gecell = crystal('Ge', [(0, 0, 0)], spacegroup=227, cell=a, primitive_cell=False)
     #check how your cell looks like
    gecell.positions[0][2]=gecell.positions[0][2]+0.00
    write('s.cif', gecell)
    structure = Struc(ase2struc(gecell))
    return structure


def compute_energy(alat, nk, ecut, a):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    potname = 'Ge.pz-bhs.UPF'
    pseudopath = os.environ['ESPRESSO_PSEUDO']
    potpath = os.path.join(pseudopath, potname)
    pseudopots = {'Ge': PseudoPotential(name=potname, path=potpath, ptype='uspp', element='Ge', functional='LDA')}
    struc = make_struc(alat=alat, a=a)
    kpts = Kpoints(gridsize=[nk, nk, nk], option='automatic', offset=False)
    runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab2/Problem8/", str(ecut)+'.'+str(nk)+'.'+str(alat)))
    input_params = PWscf_inparam({
        'CONTROL': {
            'calculation': 'scf',
            'pseudo_dir': pseudopath,
            'outdir': runpath.path,
            'tstress': True,
            'tprnfor': True,
            'disk_io': 'low',
        },
        'SYSTEM': {
            'ecutwfc': ecut,
            'nosym': True,
             },
        'ELECTRONS': {
            'diagonalization': 'david',
            'mixing_beta': 0.5,
            'conv_thr': 1e-7,
        },
        'IONS': {

        },
        'CELL': {

        },

        })

    output_file = run_qe_pwscf(runpath=runpath, struc=struc,  pseudopots=pseudopots,
                               params=input_params, kpoints=kpts)
    output = parse_qe_pwscf_output(outfile=output_file)
    return output

''''K_POINTS': {
            '{tpiba_b}'
            '6'
            '0 0 0 1'
            '0.375 0.375 0.375 1'
            '0.5 0.5 0.5 1'
            '0.625 0.25 0.625 1'
            '0.5 0.25 0.75 1'
            '0.5 0 0.5 1'
        }'''

def lattice_scan(nk, ecut, alat, a):
    #nk = 3
    #ecut = 30
    #alat = 5.0
    output = compute_energy(alat=alat, ecut=ecut, nk=nk, a=a)
    energy = output['energy']
    kpts = output['kpoints']
    forces = output['force']
    walltime = output['walltime']
    wallt = []
    for t in walltime:
        if str.isdigit(t) or '.' in t:
            wallt.append(str(t))
    #print(energy)
    #print(wallt)
    return energy, float(''.join(wallt)), kpts, forces


if __name__ == '__main__':
    # put here the function that you actually want to run
    Q = 8

    if Q == 1:
        nk = 4
        energy=[]
        ediff = []
        caltime = []
        forces = []
        alat = 5.0
        for ecut in range(5,85,5):
            energy.append(lattice_scan(nk=nk, ecut=ecut, alat=alat)[0]/(2*13.605698066))
            caltime.append(lattice_scan(nk=nk, ecut=ecut, alat=alat)[1])
            forces.append(lattice_scan(nk=nk, ecut=ecut, alat=alat)[3])
        for en in range(0,len(energy)-1):
            ediff.append(energy[len(energy)-1]-energy[en])

        plt.plot(range(5, 85, 5), energy)
        plt.xlabel('Cutoff Energy (Ry)')
        plt.ylabel('Energy (Ry/atom)')
        axes = plt.gca()
        #axes.set_ylim([0, 0.2])
        plt.title("Cutoff Convergence")
        plt.show()

        plt.plot(range(10, 85, 5), ediff)
        plt.xlabel('Cutoff Energy (Ry)')
        plt.ylabel('Energy Difference (Ry/atom)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("Convergence Energy Change")
        plt.show()

        plt.plot(range(5, 85, 5), caltime)
        plt.xlabel('Cutoff Energy (Ry)')
        plt.ylabel('Calculation Time (s)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("Calculation Time")
        plt.show()

    elif Q == 2:
        ecut = 30
        energy = []
        ediff = []
        caltime = []
        kpts=[]
        alat=5.0
        for nk in range(1, 10):
            energy.append(lattice_scan(nk=nk, ecut=ecut, alat=alat)[0] / (2 * 13.605698066))
            caltime.append(lattice_scan(nk=nk, ecut=ecut, alat=alat)[1])
            kpts.append(lattice_scan(nk=nk, ecut=ecut, alat=alat)[2])
        for en in range(0, len(energy) - 1):
            ediff.append(energy[len(energy)-1] - energy[en])
        plt.plot(kpts, energy)
        plt.xlabel('Unique K-Points')
        plt.ylabel('Energy (Ry/atom)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("K-Point Convergence")
        plt.show()

        plt.plot(kpts[1:len(kpts)], ediff)
        plt.xlabel('Unique K-Points')
        plt.ylabel('Convergence Energy Difference (Ry)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("Convergence Energy Change")
        plt.show()

        plt.plot(kpts, caltime)
        plt.xlabel('Unique K-Points')
        plt.ylabel('Calculation Time (s)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("Calculation Time")
        plt.show()

        plt.plot(range(1,10), energy)
        plt.xlabel('Input K-Points')
        plt.ylabel('Energy (Ry/atom)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("K-Point Convergence")
        plt.show()

        plt.plot(range(2,10), ediff)
        plt.xlabel('Input K-Points')
        plt.ylabel('Convergence Energy Difference (Ry)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("Convergence Energy Change")
        plt.show()

        plt.plot(range(1,10), caltime)
        plt.xlabel('Input K-Points')
        plt.ylabel('Calculation Time (s)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("Calculation Time")
        plt.show()

        plt.plot(range(1,10), kpts)
        plt.xlabel('Input K-Points')
        plt.ylabel('Unique K-Points')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("K-Points")
        plt.show()

    elif Q == 3:
        forces=[]
        fdiff=[]
        alat=5.0
        for ecut in range(5, 85, 5):
            forces.append(lattice_scan(nk=4, ecut=ecut, alat=alat)[3]/2)
            print(forces)

        for en in range(0,len(forces)-1):
            fdiff.append(forces[len(forces)-1]-forces[en])

        plt.plot(range(5, 85, 5), forces)
        plt.xlabel('Cutoff Energy (Ry)')
        plt.ylabel('Force (Ry/Bohr)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("Cutoff Convergence")
        plt.show()

        plt.plot(range(10, 85, 5), fdiff)
        plt.xlabel('Cutoff Energy (Ry)')
        plt.ylabel('Force Increment (Ry/Bohr)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("Cutoff Increment")
        plt.show()

    elif Q == 4:
        forces = []
        fdiff = []
        alat=5.0
        for nk in range(1,15):
            forces.append(lattice_scan(nk=nk, ecut=60, alat=alat)[3]/2)
            print(forces)

        for en in range(0, len(forces) - 1):
            fdiff.append(forces[len(forces)-1] - forces[en])

        plt.plot(range(1,15), forces)
        plt.xlabel('Input Grid Dimension')
        plt.ylabel('Force (Ry/Bohr)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("K-Point Convergence")
        plt.show()

        plt.plot(range(2,15), fdiff)
        plt.xlabel('Input Grid Dimension')
        plt.ylabel('Force Increment (Ry/Bohr)')
        axes = plt.gca()
        # axes.set_ylim([0, 0.2])
        plt.title("K-Point Increment")
        plt.show()

    elif Q == 5:
        e1=[]
        e2=[]
        alat1=(10.70*0.529177249)
        alat2=(10.75*0.529177249)
        ediff=[]
        ediff2=[]
        nk=4
        for ecut in range(5,85,5):
            e1temp=lattice_scan(nk=nk,ecut=ecut,alat=alat1)[0]/(2 * 13.605698066)
            e1.append(e1temp)
            e2temp=lattice_scan(nk=nk,ecut=ecut,alat=alat2)[0]/(2 * 13.605698066)
            e2.append(e2temp)
            ediff.append(e2temp-e1temp)

        for en in range(0, len(ediff) - 1):
            ediff2.append(ediff[len(ediff)-1] - ediff[en])

        plt.plot(range(5,85,5),ediff)
        plt.xlabel('Cutoff Energy (Ry)')
        plt.ylabel('Energy Difference (Ry/atom)')
        axes = plt.gca()
        plt.title('Energy Difference vs. Cutoff')
        plt.show()

        plt.plot(range(10, 85, 5), ediff2)
        plt.xlabel('Cutoff Energy (Ry)')
        plt.ylabel('Energy Difference Convergence (Ry/atom)')
        axes = plt.gca()
        plt.title('Energy Difference Convergence vs. Cutoff')
        plt.show()

    elif Q == 7:
        nk=11
        ecut=35
        energy=[]
        for alat in np.linspace(10,11.1,10):
            alat=alat*0.529177249
            energy.append(lattice_scan(nk=nk,ecut=ecut,alat=alat)[0]/(2 * 13.605698066))

        plt.plot(np.linspace(10,11.1,10),energy)
        plt.xlabel('Lattice Parameter (Bohr)')
        plt.ylabel('Energy (Ry/atom)')
        plt.title('Lattice Parameter Convergence')
        plt.show()

        ind_min = np.argmin(energy)
        alat_min=0.529177249*np.linspace(10,11.1,10)[ind_min]
        V_min=alat_min**3/(4)
        alatf2=(4*(V_min+2))**(1/3)
        alatf1=(4*(V_min+1))**(1/3)
        alatfn1=(4*(V_min-1))**(1/3)
        alatfn2=(4*(V_min-2))**(1/3)
        f2=lattice_scan(nk=nk,ecut=ecut,alat=alatf2)[0]
        f1=lattice_scan(nk=nk,ecut=ecut,alat=alatf1)[0]
        f0=(2 * 13.605698066)*energy[ind_min]
        fn1=lattice_scan(nk=nk,ecut=ecut,alat=alatfn1)[0]
        fn2=lattice_scan(nk=nk,ecut=ecut,alat=alatfn2)[0]

        d2EdV2=(-f2+16*f1-30*f0+16*fn1-fn2)/12
        B = V_min*((10**(10))**3*(1.6*10**(-19)))*d2EdV2
        print(alat_min)
        print(V_min)
        print(f2)
        print(f1)
        print(f0)
        print(fn1)
        print(fn2)
        print(B)

    elif Q == 8:
        x=0.00
        alat=5.61
        e1=0
        e2=0
        e4=0
        e5=0
        e6=x
        e3=(x)**2/(4-x**2)
        e=np.array([[e1,e6/2,e5/2],[e6/2,e2,e4/2],[e5/2,e4/2,e3]])
        a1=[0,alat,alat]
        a2=[alat,0,alat]
        a3=[alat,alat,0]
        a=np.array([a1,a2,a3])
        a=a/2
        print(a)
        a2=a
        at1=a[1]-a[2]-a[0]
        at2=a[2]-a[0]-a[1]
        at3=a[0]-a[1]-a[2]
        a=np.array([at1,at2,at3])
        e=e+np.identity(3)
        print(e)
        anew=np.dot(a2,e)
        print(anew)
        an1=anew[1]-anew[2]-anew[0]
        an2=anew[2]-anew[0]-anew[1]
        an3=anew[0]-anew[1]-anew[2]
        an=np.array([an1,an2,an3])

        print(a)
        print(an)
        E1=lattice_scan(nk=11,ecut=35,alat=5.61, a=a)[0]
        E2=lattice_scan(nk=11,ecut=35,alat=5.61, a=a)[0]

    elif Q == 9:
        lattice_scan(nk=11,ecut=35,alat=5.61)