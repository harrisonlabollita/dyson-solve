from triqs.utility import mpi 

from bethe import run_bethe




if __name__ == "__main__":
    lambdas   =  [20, 40, 50, 100, 200, 400]
    mc_cycles =  [1e5, 1e6, 1e7, 1e8, 1e9]
    n_taus    =  [5001, 7501, 10001, 20001, 50001] 

    for lamb in lambdas:
        for mc in mc_cycles:
            for n_tau in n_taus:
                if mpi.is_master_node(): print(f"running bethe for lambda = {lamb},  mc_cycles = {mc}, n_tau = {n_tau}...")
                run_bethe(lamb, n_tau, mc)
