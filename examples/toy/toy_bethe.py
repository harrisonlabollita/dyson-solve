#!/usr/bin/env python3
import sys, os

import numpy as np

import matplotlib.pyplot as plt
try: plt.style.use('publish')
except: pass

sys.path.append('../../')
from dyson_solve import Dyson

from scipy.integrate import quad

from pydlr import kernel

from triqs.gf import GfImTime, Fourier, GfImFreq, inverse, make_zero_tail


if False:
    def integral_test():
        func = lambda x : np.sqrt(1-x**2)
        f, g = quad(func, -1, 1)
        return abs(f-np.pi/2)
    print("[info] testing scipy adaptive quad", integral_test())


# evaluators
def iw_kernel(iw, omega):
    return 1.0/(iw[:,None] + omega[None, :])

def eval_G_iw(iw, beta):
    I = lambda x : +(2.0/np.pi)*iw_kernel(np.array([iw]), np.array([x]))[0,0]*np.sqrt(1-x**2)
    G, res = quad(I, -1, 1, complex_func=True)
    return G

def eval_G_tau(tau, beta):
    I = lambda x : -(2.0/np.pi)*kernel(np.array([tau])/beta, beta*np.array([x]))[0,0]*np.sqrt(1-x**2)
    G, res = quad(I, -1, 1)
    return G

def eval_G0_iw(iw, beta):
    return +1.0*iw_kernel(np.array([iw]), np.array([0]))[0,0]

def eval_G0_tau(tau, beta):
    eps = 0.0
    G = np.zeros((len(tau),1,1), dtype=complex)
    for it, t in enumerate(tau):
        G[it] = -np.exp((beta*(eps < 0) - t) * eps) / (1. + np.exp(-beta * abs(eps)))
    return G.flatten().real

eval_G_tau, eval_G_iw = np.vectorize(eval_G_tau), np.vectorize(eval_G_iw)


# set up problem
beta = 100

lamb=100

G0_iw_ref = GfImFreq(n_points=1666, beta = beta, indices=[1])
iw_array = np.array([complex(x) for x in G0_iw_ref.mesh])
G0_iw_ref.data[:] = eval_G0_iw(iw_array, beta).reshape(-1,1,1)

G_iw_ref = GfImFreq(n_points=1666, beta = beta, indices=[1])
G_iw_ref.data[:] = eval_G_iw(iw_array, beta).reshape(-1,1,1)

Sigma_iw_ref = 0.25 * G_iw_ref.data.flatten()

print('computing G(τ)')
G_tau_ref = GfImTime(n_points=10000, beta = beta, indices=[1])
tau_mesh = np.array([float(x) for x in G_tau_ref.mesh])

if os.path.isfile('G_tau_ref.txt'):
    data = np.loadtxt('G_tau_ref.txt', dtype=complex)
else:
    data = eval_G_tau(tau_mesh, beta)
    print('saving ref data', np.savetxt('G_tau_ref.txt', data))

G_tau_ref.data[:] = data.reshape(-1,1,1)


def experiment(tol):
    print('running an experiment with tol = ', tol)
    dys = Dyson(lamb, eps=tol, options=dict(
                                            maxiter=5000, 
											gtol=tol*0.001 if tol < 1e-8 else 1e-8,
                                            xtol=tol*0.001 if tol < 1e-8 else 1e-8,
                                            finite_diff_rel_step=1e-14
                                            )
                                            )
    print(dys)

    print('gettting G(τ)+η')
    G_tau = GfImTime(n_points=10000, beta = beta, indices=[1])
    G_tau.data[:] += G_tau_ref.data
    # add random noise between η*(-1, 1)
    G_tau.data[:] += tol*(2*np.random.rand(*G_tau.data.shape)-1)


    # Giw from Gtau via TRIQS
    print('computing G(τ) -> G(iω)...')
    tail = make_zero_tail(G_iw_ref, 2); tail[1] = 1.0
    G_iw = G_iw_ref.copy()
    G_iw << Fourier(G_tau, known_moments=tail)
    print('done...')

    G0_tau = GfImTime(n_points=10000, beta = beta, indices=[1])
    G0_tau.data[:] = eval_G0_tau(tau_mesh, beta).reshape(-1,1,1)
    print('done...')

    # Sigma from Dyson equation Σ = G0^-1 - G^-1
    print('Σ from dyson equation')
    Sigma_iw = inverse(G0_iw_ref) - inverse(G_iw)
    print('done...')

    print('Σ from residual minimization')
    Sigma_moments = np.array([0.0, +0.25],dtype=complex).reshape(-1,1,1)

    result = dys.solve(G0_tau=G0_tau.data,
                       G_tau=G_tau.data,
                       Sigma_moments=Sigma_moments,
                       beta=beta, 
                       om_mesh=iw_array, 
                       tau_mesh=tau_mesh 
                       )

    Sigma_iw_res = result.Sigma_iw
    Sigma_iw_res = Sigma_iw_res.flatten()
    print('done...')

    return {'dyson' : Sigma_iw.data.flatten(), 'res' : Sigma_iw_res }

colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']

def add_results_to_plot(ax, results):
    assert len(ax) == 2
    c=0
    for key in results.keys():
        color=colors[c]
        dys = results[key]['dyson']
        res = results[key]['res']

        if key == 1e-4:
            ax[0].plot(iw_array.imag, Sigma_iw_ref.imag, 'o', ms=3, mfc='none', color='tab:red', label='ref')
            ax[0].plot(iw_array.imag, dys.imag, '.', ms=2, color='tab:blue', 
                    label=r'$G_{0}^{-1}-G^{-1}$')
        ax[1].loglog(iw_array.imag, np.abs(dys-Sigma_iw_ref), ls='-', color=color, label=r'$\eta=$ '+'{:1.0e}'.format(key))
        ax[1].loglog(iw_array.imag, np.abs(res-Sigma_iw_ref), ls='--', color=color)
        ax[1].set_ylabel(r'$\Sigma$ asbolute error')
        c+=1

tols = [1e-4, 1e-6, 1e-8]
results = {tol : experiment(tol) for tol in tols }
fig, ax = plt.subplots(2,1,figsize=(5, 7))
ax[0].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
ax[0].set_ylim(-0.1, 0.1)
add_results_to_plot(ax, results)
ax[0].legend(frameon=True, framealpha=0.8, facecolor='white', edgecolor='none', loc='upper right', fontsize=8)
ax[1].loglog(iw_array.imag[1700:1900], 1e-3*(iw_array.imag**2)[1700:1900], ls='dotted', lw=2, color='k')
ax[1].axhline(100, color='k', ls='-', label=r'$G_{0}^{-1}-G^{-1}$')
ax[1].axhline(100, color='k', ls='--', label='res. min')
ax[1].axhline(100, color='k', ls='dotted', label=r'$\mathcal{O}(\nu_{n}^{2})$')
ax[1].set_ylim(1e-16, 1e0)
ax[1].legend(frameon=True, framealpha=0.8, facecolor='white', edgecolor='none', ncols=2, loc='lower left', fontsize=8)
ax[1].set_xlabel(r'$i\nu_{n}$')

for a, let in zip(ax, ['(a)', '(b)']):
    t = a.text(0.03, 0.85, let, transform = a.transAxes, size=14) 
    t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))
plt.savefig('dyson_exact_bethe_abserr.pdf')
