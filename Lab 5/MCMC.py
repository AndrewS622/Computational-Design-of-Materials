from labutil.src.plugins.ising import *
import matplotlib.pyplot as plt
import numpy as np

def run_mcmcising(N,n_eq,n_mc,T):
    #### RUN MC CODE ####
    E, M, E_eq, M_eq, imagelist = mc_run(N=N,n_eq=n_eq,n_mc=n_mc,T=T)

    x_eq = np.arange(n_eq*N*N)
    x_mc = np.arange(n_mc*N*N)+(n_eq*N*N)

    return(E,M,E_eq,M_eq,x_eq,x_mc,imagelist)

def autocorr(E):
    l = len(E)
    n = 1                               # spacing of dt to test
    N = 500                             # total number of dt to test
    acorr = []
    for dt in range(1,N,n):
        m = l - max(range(1,N,n))
        s1=0
        s2=0
        for t in range(0,m):
            s1 = s1 + E[t]
            s2 = s2 + E[t]*E[t+dt]
        E1 = s1/m
        E2 = s2/m
        acorr.append((E2 - E1**2))
    # plt.plot(range(1,N,n),acorr)
    # plt.xlabel('Separation Between Points')
    # plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation Function')
    # plt.show()
    return(acorr)

def converge(E):
    Eavg = []
    countthresh = 100
    valthresh = 0.001
    Ethresh = 10**(-6)
    Eavg1 = np.mean(E)
    count = 0
    for i in range(1, len(E)):
        Eavg2 = np.mean(E[i:len(E)])
        Eavg.append(Eavg2)
        dE = []
        if abs(Eavg2 - Eavg1) < valthresh:
            count = count + 1
            dE.append(Eavg2 - Eavg1)
            if count > countthresh and abs(np.mean(dE)) < Ethresh:
                ind = i-100
                print(np.mean(dE))
                break
        else:
            count = 0
            dE = []
        Eavg1 = Eavg2
    conv = np.mean(E[ind:len(E)])
    print(ind, conv)
    print('ind, conv')
    return conv, ind


if __name__ == '__main__':
    # put here the function that you actually want to run
    Q = 3

    if Q == 1:
        Econv = []
        Eind = []
        Eindpersite = []
        Corrtime = []
        for N in range (10,40,5):
            T=2
            n_eq=800         # Average number of equilibration steps (flips) per site
            n_mc=200        # Average number of Monte Carlo steps (flips) per site
            print(N)

            [E,M,E_eq,M_eq,x_eq,x_mc,imagelist]=run_mcmcising(N=N,n_eq=n_eq,n_mc=n_mc,T=T)
            # acorr = autocorr(E)
            # thresh = next(i for i in acorr if i < 0.1)
            # Corrtime.append(thresh)

            Etot = E_eq + E
            [ec, eind] = converge(Etot)
            Econv.append(ec)
            Eind.append(eind)
            Eindpersite.append(eind / (N * N))
            # plt.plot(Etot)
            # plt.show()

            # fig, (ax_energy, ax_mag) = plt.subplots(2, 1, sharex=True, sharey=False)
            #
            # ax_energy.plot(x_eq, E_eq, label='Equilibration')
            # ax_energy.plot(x_mc, E, label='Production')
            # ax_energy.axvline(n_eq * (N ** 2), color='black')
            # ax_energy.legend()
            # ax_energy.set_ylabel('Energy per site/ J')
            #
            # ax_mag.plot(x_eq, M_eq, label='Equilibration')
            # ax_mag.plot(x_mc, M, label='Production')
            # ax_mag.axvline(n_eq * (N ** 2), color='black')
            # ax_mag.legend()
            # ax_mag.set_ylabel('Magnetization per site')
            # ax_mag.set_xlabel('Number of flip attempts')
            #
            # animator(imagelist)
            # plt.show()

        plt.plot(range(10,40,5),Eind, '.')
        plt.xlabel('System Size Index')
        plt.ylabel('Index of Converged Energy (Total Steps)')
        plt.title('Converged Energy Step vs System Size')
        plt.show()
        plt.plot(range(10,40,5),Econv, '.')
        plt.xlabel('System Size Index')
        plt.ylabel('Converged Energy')
        plt.title('Converged Energy vs System Size')
        plt.show()
        # plt.plot(range(10,40,5),Corrtime)
        # plt.xlabel('System Size Index')
        # plt.ylabel('Correlation Time')
        # plt.show()
        plt.plot(range(10, 40, 5), Eindpersite, '.')
        plt.xlabel('System Size Index')
        plt.ylabel('Index of Converged Energy (MC Steps)')
        plt.title('Converged Energy Step vs System Size')
        plt.show()

    elif Q == 2:
        Eavg=[]
        Ttest=[]
        Estd=[]
        for T in np.linspace(1,3,51):
            N=10
            n_eq=800
            n_mc=500
            Ttest.append(T)
            print(T)
            [E,M,E_eq,M_eq,x_eq,x_mc,imagelist]=run_mcmcising(N=N,n_eq=n_eq,n_mc=n_mc,T=T)

            Esample = E[0:len(E):N*N]
            print(len(Esample))
            Eavg.append(np.mean(Esample))
            Estd.append(np.std(Esample))
            # plt.plot(E_eq[0:len(E_eq):N*N] + E[0:len(E):N*N])
            # plt.show()
        Cv = []
        Cv2 = []
        dT = Ttest[1] - Ttest[0]
        for i in range(0,len(Eavg)-1):
            dE = Eavg[i+1]-Eavg[i]
            Cv.append(dE/dT)
        for i in range(0,len(Estd)):
            Cv2.append(((Estd[i]**2)/Ttest[i]**2))
        Tnew = Ttest[0:(len(Ttest)-1)]+dT/2
        plt.plot(Tnew,Cv)
        plt.xlabel('Temperature')
        plt.ylabel('Specific Heat')
        plt.title('Specific Heat from Derivative Estimate N = 10')
        plt.show()
        plt.plot(Ttest,Cv2)
        plt.xlabel('Temperature')
        plt.ylabel('Specific Heat')
        plt.title('Specific Heat from Fluctuation Dissipation N = 10')
        plt.show()
        plt.plot(Ttest,Eavg)
        plt.xlabel('Temperature')
        plt.ylabel('Average Energy')
        plt.show()

    elif Q == 3:
        Mconv = []
        Mind = []
        Mindpersite = []
        Corrtime = []
        for N in range(10, 40, 5):
            n_eq = 800  # Average number of equilibration steps (flips) per site
            n_mc = 500  # Average number of Monte Carlo steps (flips) per site
            print(N)

            [E, M, E_eq, M_eq, x_eq, x_mc, imagelist] = run_mcmcising(N=N, n_eq=n_eq, n_mc=n_mc, T=T)
            # acorr = autocorr(M)
            # thresh = next(i for i in acorr if i < 0.1)
            # Corrtime.append(thresh)

            Mtot = M_eq + M
            [mc, mind] = converge(Mtot)
            Mconv.append(mc)
            Mind.append(mind)
            Mindpersite.append(mind / (N * N))

            fig, (ax_energy, ax_mag) = plt.subplots(2, 1, sharex=True, sharey=False)

            ax_energy.plot(x_eq, E_eq, label='Equilibration')
            ax_energy.plot(x_mc, E, label='Production')
            ax_energy.axvline(n_eq * (N ** 2), color='black')
            ax_energy.legend()
            ax_energy.set_ylabel('Energy per site/ J')

            ax_mag.plot(x_eq, M_eq, label='Equilibration')
            ax_mag.plot(x_mc, M, label='Production')
            ax_mag.axvline(n_eq * (N ** 2), color='black')
            ax_mag.legend()
            ax_mag.set_ylabel('Magnetization per site')
            ax_mag.set_xlabel('Number of flip attempts')

            animator(imagelist)
            plt.show()

        plt.plot(range(10, 40, 5), Mind, '.')
        plt.xlabel('System Size Index')
        plt.ylabel('Index of Converged Magnetization (Total Steps)')
        plt.title('Converged Magnetization Step vs. System Size')
        plt.show()
        plt.plot(range(10, 40, 5), Mconv, '.')
        plt.xlabel('System Size Index')
        plt.ylabel('Converged Magnetization')
        plt.title('Converged Magnetization')
        plt.show()
        # plt.plot(range(10, 40, 5), Corrtime)
        # plt.xlabel('System Size Index')
        # plt.ylabel('Correlation Time')
        # plt.show()
        plt.plot(range(10, 40, 5), Mindpersite, '.')
        plt.xlabel('System Size Index')
        plt.ylabel('Index of Converged Magnetization(MC Steps)')
        plt.title('Converged Magnetization Step per Site vs. System Size')
        plt.show()

    elif Q == 4:
        Mavg = []
        Ttest = []
        Mstd = []
        N=40
        for T in np.linspace(1,3,51):
            n_eq = 800
            n_mc = 500
            Ttest.append(T)
            print(T)
            [E, M, E_eq, M_eq, x_eq, x_mc, imagelist] = run_mcmcising(N=N, n_eq=n_eq, n_mc=n_mc, T=T)

            Mavg.append(np.mean(M[0:len(M)]))
            Mstd.append(np.std(M[0:len(M)]))
        X = []
        print(Mavg)
        for i in range(0, len(Mstd)):
            X.append((Mstd[i]**2 / Ttest[i]))

        plt.plot(Ttest, X)
        plt.xlabel('Temperature')
        plt.ylabel('Susceptibility')
        plt.title('Susceptibility')
        plt.show()

        plt.plot(Ttest, Mavg)
        plt.xlabel('Temperature')
        plt.ylabel('Average Magnetization')
        plt.title('Average Magnetization vs. Temperature')
        plt.show()