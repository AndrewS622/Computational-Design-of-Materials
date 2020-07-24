from labutil.src.plugins.ising import *
import matplotlib.pyplot as plt

def run_mcmcising(N,n_eq,n_mc,T)
    #### RUN MC CODE ####
    E, M, E_eq, M_eq, imagelist = mc_run(N=N,n_eq=n_eq,n_mc=n_mc,T=T)

    x_eq = np.arange(n_eq*N*N)
    x_mc = np.arange(n_mc*N*N)+(n_eq*N*N)

    return(E,M,E_eq,M_eq,x_eq,x_mc,imagelist)


if __name__ == '__main__':
    # put here the function that you actually want to run

    for N in range (2,20,2):
        T=2
        n_eq=75         # Average number of equilibriation steps (flips) per site
        n_mc=100        # Average number of Monte Carlo steps (flips) per site

        [E,M,E_eq,M_eq,x_eq,x_mc,imagelist]=run_mcmcising(N=N,n_eq=n_eq,n_mc=n_mc,T=T)


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
