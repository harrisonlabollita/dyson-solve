import sys, os
from triqs.utility import mpi 

setenv  = lambda x, default : os.getenv(x) if os.getenv(x) is not None else default

RUN = setenv('RUN', 0)

get_reference=False
if not get_reference:
    from bethe import run_bethe
    lambdas   =  [1, 2, 4, 6, 8, 10, 12, 15, 20, 22, 25, 27, 30,40, 50]
    mc_cycles =  [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
    n_taus    =  [10001] 
else:
    from ref_bethe import run_bethe

if __name__ == "__main__":

    if RUN:
        if not get_reference:
            for lamb in lambdas:
                for mc in mc_cycles:
                    for n_tau in n_taus:
                        if mpi.is_master_node(): print(f"running bethe for lambda = {lamb},  mc_cycles = {int(mc)}, n_tau = {n_tau}...")
                        run_bethe(lamb, n_tau, mc)
        else:
            if mpi.is_master_node(): print(f"running bethe to get reference data")
            run_bethe(0, 10001, 1e11)
    else:
        msg="""this data takes a very long time to generate. You should be running this on HPC.
               if you are sure that you would like to continue, then you can run this script by setting
               your environment variable 'RUN'. For example, from the command line execute,
               >>> RUN=1; mpirun python3 run_example.py
            """
        print(msg)

    
