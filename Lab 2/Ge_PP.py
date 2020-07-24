def input
    runpath = Dir(
        path=os.path.join(os.environ['WORKDIR'], "Lab2/Problem9/", str(ecut) + '.' + str(nk) + '.' + str(alat)))
    input_params = PWscf_inparam({
        'INPUTPP': {
            'prefix': 'pwscf',
            'outdir': runpath.path,
            'filband': 'Ge_bands.dat',
        },
        'SYSTEM': {
            'ecutwfc': ecut,
            'nbnd': 5
        },
    })

if __name__ == '__main__':
# put here the function that you actually want to run

    input()