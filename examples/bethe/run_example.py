from bethe import run_bethe
from dyson_solve import Dyson

import numpy as np


lambdas   =  [20, 40, 50, 100, 200, 400]
mc_cycles =  [1e5, 1e6, 1e7, 1e8, 1e9]
n_taus    =  [5001, 7501, 10001, 20001, 50001] 

for lamb in lambds:
    for mc in mc_cycles:
        for n_tau in n_taus:
            run_bethe(lamb, n_tau, mc)
