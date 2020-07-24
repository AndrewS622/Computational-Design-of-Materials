from my_labutil.src.plugins.lammps import *
from ase.spacegroup import crystal
from ase.build import make_supercell
import matplotlib.pyplot as plt
import numpy as np


def make_struc(size):
    """
    Creates the crystal structure using ASE.
    :param size: supercell multiplier
    :return: structure object converted from ase
    """
    alat = 4.10
    unitcell = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[alat, alat, alat, 90, 90, 90])
    multiplier = numpy.identity(3) * size
    supercell = make_supercell(unitcell, multiplier)
    structure = Struc(ase2struc(supercell))
    return structure


def compute_dynamics(size, timestep, nsteps, temperature, ensemble):
    """
    Make an input template and select potential and structure, and input parameters.
    Return a pair of output file and RDF file written to the runpath directory.
    """
    if ensemble == 'nve':
        intemplate = """
        # ---------- Initialize simulation ---------------------
        units metal
        atom_style atomic
        dimension  3
        boundary   p p p
        read_data $DATAINPUT

        pair_style eam/alloy
        pair_coeff * * $POTENTIAL  Al

        velocity  all create $TEMPERATURE 87287 dist gaussian

        # ---------- Describe computed properties------------------
        compute msdall all msd
        thermo_style custom step pe ke etotal temp press density c_msdall[4]
        thermo $TOUTPUT

        # ---------- Specify ensemble  ---------------------
        fix  1 all nve
        #fix  1 all nvt temp $TEMPERATURE $TEMPERATURE $TDAMP

        # --------- Compute RDF ---------------
        compute rdfall all rdf 100 1 1
        fix 2 all ave/time 1 $RDFFRAME $RDFFRAME c_rdfall[*] file $RDFFILE mode vector

        # --------- Run -------------
        timestep $TIMESTEP
        run $NSTEPS
        """

    elif ensemble == 'nvt':
        intemplate = """
        # ---------- Initialize simulation ---------------------
        units metal
        atom_style atomic
        dimension  3
        boundary   p p p
        read_data $DATAINPUT

        pair_style eam/alloy
        pair_coeff * * $POTENTIAL  Al

        velocity  all create $TEMPERATURE 87287 dist gaussian

        # ---------- Describe computed properties------------------
        compute msdall all msd
        thermo_style custom step pe ke etotal temp press density c_msdall[4]
        thermo $TOUTPUT

        # ---------- Specify ensemble  ---------------------
        #fix  1 all nve
        fix  1 all nvt temp $TEMPERATURE $TEMPERATURE $TDAMP

        # --------- Compute RDF ---------------
        compute rdfall all rdf 100 1 1
        fix 2 all ave/time 1 $RDFFRAME $RDFFRAME c_rdfall[*] file $RDFFILE mode vector

        # --------- Run -------------
        timestep $TIMESTEP
        run $NSTEPS
        """

    potential = ClassicalPotential(ptype='eam', element='Al', name='Al_zhou.eam.alloy')
    runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab4/Problem2/Melt",str(size) + "_" + str(timestep) + "_" + str(temperature)))
    struc = make_struc(size=size)
    inparam = {
        'TEMPERATURE': temperature,
        'NSTEPS': nsteps,
        'TIMESTEP': timestep,
        'TOUTPUT': 100,                 # how often to write thermo output
        'TDAMP': 50 * timestep,       # thermostat damping time scale
        'RDFFRAME': int(nsteps / 4),   # frames for radial distribution function
    }
    outfile = lammps_run(struc=struc, runpath=runpath, potential=potential,
                                  intemplate=intemplate, inparam=inparam)
    output = parse_lammps_thermo(outfile=outfile)
    rdffile = get_rdf(runpath=runpath)
    rdfs = parse_lammps_rdf(rdffile=rdffile)
    return output, rdfs


def md_run(size,timestep,nsteps,temperature,ensemble):
    output, rdfs = compute_dynamics(size=size, timestep=timestep, nsteps=nsteps, temperature=temperature, ensemble=ensemble)
    [simtime, pe, ke, energy, temp, press, dens, msd] = output
    ## ------- plot output properties
    return energy,simtime,temp,pe,ke,press,dens,msd,rdfs

    # ----- plot radial distribution functions
    #for rdf in rdfs:
        #plt.plot(rdf[0], rdf[1])
    #plt.show()


if __name__ == '__main__':
    # put here the function that you actually want to run
    Q=7
    if Q <4:
        ensemble = 'nve'
    else:
        ensemble = 'nvt'

    Econv = []
    if Q == 1:
        # 1 is used for timestep optimization (1A)
        Etrend=[]
        for timestep in np.linspace(0.001,0.02,20):
            E = []
            nsteps=int(10/timestep)
            [E, simtime, T, PE, KE, p, d, msd] = md_run(size=3,timestep=timestep,nsteps=10000,temperature=300,ensemble=ensemble)

            Eavg = []
            E=np.array(E).astype(np.float)
            for i in range(1,len(E)):
                print(E[(i-1):(len(E)-1)])
                Eavg.append(np.mean(E[(i-1):(len(E)-1)]))
            Eavg.append(E[len(E)-1])

            m = np.median(Eavg)
            print(m)
            s = np.std(Eavg)
            print(s)
            ind = next(i for i in Eavg if abs(i) < (abs(m)+2*s))
            print(ind)
            Etrend.append(ind)

            time = np.array(simtime,dtype=float)*timestep

            # plt.plot(time, T)
            # plt.show()

            if timestep == 0.02:
                plt.plot(time, E)
                plt.xlabel('Time (ps)')
                plt.ylabel('Energy')
                plt.title("Energy vs. Time")
                plt.show()
            #
            # plt.plot(time, Eavg)
            # plt.xlabel('Time of First Included Value in Average (ps)')
            # plt.ylabel('Mean Energy')
            # plt.title("Mean Energy vs. Time")
            # plt.show()

        plt.plot(np.linspace(0.001,0.02,20), Etrend)
        plt.xlabel('Timestep (ps)')
        plt.ylabel('Mean Energy')
        plt.title("Timestep Dependence of Energy")
        plt.show()

    elif Q == 2:
        # 2 is used for temperature vs. time plots at multiple T (1B)
        E=[]
        T=[]
        for temp in np.linspace(200,500,7):
            [E, simtime, T, PE, KE, p, d, msd] = md_run(size=3, timestep=0.001, nsteps=10000, temperature=temp, ensemble=ensemble)

            time = np.array(simtime, dtype=float) * 0.001

            plt.figure(1)
            plt.plot(time, T)
            plt.xlabel('Time (ps)')
            plt.ylabel('Temperature (K)')
            plt.title("Temperature Evolution")

            plt.figure(2)
            plt.plot(time, PE)
            plt.xlabel('Time (ps)')
            plt.ylabel('Potential Energy')
            plt.title("Potential Energy Evolution")

            plt.figure(3)
            plt.plot(time, KE)
            plt.xlabel('Time (ps)')
            plt.ylabel('Kinetic Energy')
            plt.title("Kinetic Energy Evolution")

        plt.figure(1)
        plt.show()

        plt.figure(2)
        plt.show()

        plt.figure(3)
        plt.show()

    elif Q == 3:
        # 3 is used for supercell dependence (1C)
        E=[]
        T=[]
        Tfluc=[]
        siz=[]
        for size in range(3,15,1):
            Tdiff=[]
            #for temp in np.linspace(200, 500, 20):
            [E, simtime, T, PE, KE, p, d, msd] = md_run(size=size, timestep=0.001, nsteps=5000, temperature=300, ensemble=ensemble)

            siz.append(1/(4*size**3))
            time = np.array(simtime, dtype=float) * 0.001
            plt.plot(time, T)
            T = np.array(T).astype(np.float)

            m = np.median(T)
            s = np.std(T)
            conv = next(i for i in T if abs(i) < (abs(m) + 2 * s))

            ind = np.nonzero(T==conv)[0][0]
            Tm=np.mean(T[ind:(len(T) - 1)])

            for i in range(ind,len(T)-1):
                Tdiff.append((T[i]-Tm)**2)
            Tfluc.append(np.mean(Tdiff))

        plt.xlabel('Time (ps)')
        plt.ylabel('Temperature (K)')
        plt.title("Temperature Evolution")
        plt.show()

        plt.plot(siz,Tfluc)
        plt.xlabel('Reciprocal Size')
        plt.ylabel('Average Temperature Fluctuation')
        plt.title('Temperature Fluctuation vs. System Size')
        plt.show()

    elif Q == 4:
        # 4 is used for finding relevant T for melting
        E = []
        T = []
        for temp in np.linspace(200, 1300, 12):
            [E, simtime, T, PE, KE, p, d, msd] = md_run(size=3, timestep=0.001, nsteps=20000, temperature=temp, ensemble=ensemble)

            time = np.array(simtime, dtype=float) * 0.001

            plt.plot(time, E)
        plt.xlabel('Time (ps)')
        plt.ylabel('Energy')
        plt.title("Energy Evolution")
        plt.show()

    elif Q == 5:
        # 5 is used to perform timestep optimization in a regime below melting
        Etrend=[]
        for timestep in np.linspace(0.001,0.02,20):
            E = []

            [E, simtime, T, PE, KE, p, d, msd] = md_run(size=3,timestep=timestep,nsteps=10000,temperature=500,ensemble=ensemble)

            Eavg = []
            E = np.array(E).astype(np.float)
            for i in range(1, len(E)):
                print(E[(i - 1):(len(E) - 1)])
                Eavg.append(np.mean(E[(i - 1):(len(E) - 1)]))
            Eavg.append(E[len(E) - 1])

            m = np.median(Eavg)
            print(m)
            s = np.std(Eavg)
            print(s)
            ind = next(i for i in Eavg if abs(i) < (abs(m) + 2 * s))
            print(ind)
            Etrend.append(ind)

            time = np.array(simtime, dtype=float) * timestep

            # plt.plot(time, T)
            # plt.show()

            if timestep == 0.02:
                plt.plot(time, E)
                plt.xlabel('Time (ps)')
                plt.ylabel('Energy')
                plt.title("Energy vs. Time")
                plt.show()
                #
                # plt.plot(time, Eavg)
                # plt.xlabel('Time of First Included Value in Average (ps)')
                # plt.ylabel('Mean Energy')
                # plt.title("Mean Energy vs. Time")
                # plt.show()

        plt.plot(np.linspace(0.001, 0.02, 20), Etrend)
        plt.xlabel('Timestep (ps)')
        plt.ylabel('Mean Energy')
        plt.title("Timestep Dependence of Energy")
        plt.show()

    elif Q == 6:
        # 6 is used to perform supercell optimization in a regime below melting
        E = []
        T = []
        for size in range(3, 13, 1):
            [E, simtime, T, PE, KE, p, d, msd] = md_run(size=size, timestep=0.001, nsteps=5000, temperature=500, ensemble=ensemble)

            time = np.array(simtime, dtype=float) * 0.001
            plt.plot(time, T)

        plt.xlabel('Time (ps)')
        plt.ylabel('Temperature (K)')
        plt.title("Temperature Evolution")
        plt.show()

    elif Q == 7:
        # 7 is used for the NVT ensemble with the results from above
        E=[]
        for temp in np.linspace(1050,1150,6):
            [E, simtime, T, PE, KE, p, d, msd,rdfs] = md_run(size = 7, timestep = 0.001, nsteps = 10000, temperature = temp, ensemble = ensemble)
            print(simtime)
            time = np.array(simtime, dtype=float) * 0.001

            plt.figure(1)
            plt.plot(time, E)
            plt.xlabel('Time (ps)')
            plt.ylabel('Energy')
            plt.title("Energy Evolution")

            plt.figure(2)
            plt.plot(time, msd)
            plt.xlabel('Time (ps)')
            plt.ylabel('Mean Square Density')
            plt.title("Mean Square Density Evolution")

            plt.figure(3)
            rdfs = np.array(rdfs).astype(np.float)
            rdfm=[]
            print(rdfs.shape)
            s = rdfs.shape[0]
            for i in range(0,rdfs.shape[2]):
                sum=0
                for j in range(0,s):
                    sum = sum + rdfs[j][1][i]
                rdfm.append(sum/s)
            plt.plot(rdfs[0][0], rdfm)
            plt.title("Radial Density Function")

        plt.figure(1)
        plt.show()

        plt.figure(2)
        plt.show()

        plt.figure(3)
        plt.show()
