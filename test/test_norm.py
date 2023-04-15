#!/usr/bin/env python3

import sys
import numpy as np
from triqs.gf import *
from pydlr import kernel, dlr

sys.path.append('../')
from dyson_solve import Dyson

beta = 100
eps  = 1

dys = Dyson(lamb=100, eps=1e-14)

# G(τ) = exp(-τϵ)/( 1 + exp(-βϵ))
analytic_norm = np.sqrt(np.tanh(0.5*beta*eps)/(2*eps*beta))

tau_l = dys.d.get_tau(beta)
G = np.zeros((len(tau_l),1,1), dtype=complex)
for it, tau in enumerate(tau_l): G[it] = -np.exp((beta*(eps<0)-tau)*eps) / (1. + np.exp(-beta * abs(eps)))

g_xaa = dys.d.dlr_from_tau(G)

dlr_norm = np.sqrt(g_xaa.flatten().T@dys.Mkl@g_xaa.flatten())

print("||G|| (ana) = ", analytic_norm)
print('||G|| (dlr) = ', dlr_norm.real)
print(f'Δ||G|| = {abs(analytic_norm-dlr_norm.real):.10e}')
