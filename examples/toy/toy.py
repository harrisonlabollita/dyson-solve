import sys

from triqs.gf import *

import numpy as np
np.random.seed(85281)

import matplotlib.pyplot as plt
plt.style.use('publish')

sys.path.append('../../')
from dyson_solve import Dyson


def eval_G0_tau(tau, U, beta):
    eps = -U/2
    G = np.zeros((len(tau),1,1), dtype=complex)
    for it, t in enumerate(tau):
        G[it] = -np.exp((beta*(eps < 0) - t) * eps) / (1. + np.exp(-beta * abs(eps)))
    return G

def eval_G_tau(tau, U, beta):
    eps1=U/2
    eps2=-U/2
    G = np.zeros((len(tau),1,1),dtype=complex)
    for it, t in enumerate(tau):
        G[it] = -0.5*np.exp((beta*(eps1<0) - t) * eps1) / (1. + np.exp(-beta * eps1)) \
                -0.5*np.exp((beta*(eps2<0) - t) * eps2) / (1. + np.exp(-beta * abs(eps2)))
    return G

# set up problem
U = 2.0
beta = 100

lamb = 100
eps  = 1e-17

tol = 1e-10

iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=10000)

iw_array = np.array([complex(x) for x in iw_mesh])

Sigma_iw_ref = Gf(mesh=iw_mesh, target_shape=[1,1])
Sigma_iw_ref << U/2 + 0.25*U*U*inverse(iOmega_n)

dys = Dyson(lamb=lamb, eps=eps, 
            options=dict(maxiter=4000, 
                         disp=True, 
                         gtol=1e-32, 
                         xtol=1e-100, 
                         finite_diff_rel_step=1e-20
                         ))

tau_l = dys.d.get_tau(beta)

G0_tau = eval_G0_tau(tau_l, U, beta)
#G0_tau += np.random.normal(scale=tol, size=G0_tau.shape)

G_tau  = eval_G_tau(tau_l, U, beta)
#G_tau += np.random.normal(scale=tol, size=G_tau.shape)

G0_dlr = dys.d.dlr_from_tau(G0_tau)
G_dlr = dys.d.dlr_from_tau(G_tau)

G0_iw = dys.d.eval_dlr_freq(G0_dlr, iw_array, beta)
G_iw = dys.d.eval_dlr_freq(G_dlr, iw_array, beta)

Sigma_iw_dyson = np.linalg.inv(G0_iw) - np.linalg.inv(G_iw)

Sigma_moments = np.array([U/2, 0.25*U*U],dtype=complex).reshape(-1,1,1)

result = dys.solve(G0_tau=G0_tau, 
                   G_tau=G_tau, 
                   Sigma_moments=Sigma_moments,
                   beta=beta, 
                   om_mesh=iw_array)

Sigma_iw_res = result.Sigma_iw


# plot results
plt.figure()
plt.loglog(iw_array.imag, np.abs(Sigma_iw_dyson.flatten()-Sigma_iw_ref.data.flatten()), 
           label=r'$G_{0}^{-1}-G^{-1}$')
plt.loglog(iw_array.imag, np.abs(Sigma_iw_res.flatten()-Sigma_iw_ref.data.flatten()),
           label='res. min.')
plt.legend()
plt.xlabel(r'$\nu_{n}$');  plt.ylabel(r'$|\Sigma(i\nu_{n})-\Sigma_{\mathrm{exact}}(i\nu_{n})|$')
plt.show()
