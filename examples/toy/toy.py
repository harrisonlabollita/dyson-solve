from triqs.gf import *

import numpy as np
np.random.seed(85281)

import matplotlib.pyplot as plt

from toy_dyson_solve import Dyson


def G0_tau(tau, U, beta):
    eps = -U/2
    G = np.zeros((len(tau),1,1), dtype=complex)
    for it, t in enumerate(tau):
        G[it,:,:] = -np.exp((beta*(eps < 0) - t) * eps) / (1. + np.exp(-beta * abs(eps)))
    return G

def G_tau(tau, U, beta):
    eps1=U/2
    eps2=-U/2
    G = np.zeros((len(tau),1,1),dtype=complex)
    for it, t in enumerate(tau):
        G[it] = -0.5*np.exp((beta*(eps1<0) - t) * eps1) / (1. + np.exp(-beta * eps1)) \
                -0.5*np.exp((beta*(eps2<0) - t) * eps2) / (1. + np.exp(-beta * abs(eps2)))
    return G

U = 2.0
beta = 100

lamb = 100
eps  = 1e-14

tol = 1e-10

iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=1000)

G0_iw = Gf(mesh=iw_mesh, target_shape=[1,1])
G0_iw << inverse(iOmega_n + U/2)

G0_tau_triqs = make_gf_from_fourier(G0_iw)

G_iw  = Gf(mesh=iw_mesh, target_shape=[1,1])
G_iw  << inverse(2*(iOmega_n + U/2)) + inverse(2*(iOmega_n - U/2))

G_tau_triqs = make_gf_from_fourier(G_iw)
G_tau_triqs.data[:]
tau_i = np.array([float(x) for x in G_tau_triqs.mesh])

Sigma_iw_mach = inverse(G0_iw) - inverse(G_iw)

#G_tau_qmc = G_tau.copy()
#G_tau_qmc.data[:] += np.random.normal(scale=tol, size=G_tau_qmc.data.shape)
#G_iw_qmc = make_gf_from_fourier(G_tau_qmc)

Sigma_iw_ref = G_iw.copy()
Sigma_iw_ref << U/2 + 0.25*U*U*inverse(iOmega_n)

dys = Dyson(lamb=lamb, eps=eps, 
            options=dict(maxiter=400, disp=True, gtol=1e-32, xtol=1e-10, finite_diff_rel_step=1e-20))

tau_l = dys.d.get_tau(beta)

G0 = G0_tau(tau_l, U, beta)
G0 += np.random.normal(scale=tol, size=G0.shape)
G  = G_tau(tau_l, U, beta)
G += np.random.normal(scale=tol, size=G0.shape)


Sigma_moments = np.array([U/2, 0.25*U*U],dtype=complex).reshape(-1,1,1)

sig_xaa, sol = dys.constrained_lstsq_dlr_from_tau(tau_l, G, G0, beta, Sigma_moments)

om_mesh = np.array([complex(x) for x in G_iw.mesh])

Sigma_iw_res = dys.d.eval_dlr_freq(sig_xaa, om_mesh, beta)
Sigma_iw_res += Sigma_moments[0]

plt.figure()
plt.loglog(om_mesh.imag, np.abs(Sigma_iw_mach.data.flatten()-Sigma_iw_ref.data.flatten()))
plt.loglog(om_mesh.imag, np.abs(Sigma_iw_res.flatten()-Sigma_iw_ref.data.flatten()))
plt.show()
