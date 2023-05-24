#!/usr/bin/env python3
import sys, os

import numpy as np
np.random.seed(1)

import matplotlib.pyplot as plt
try: plt.style.use('publish')
except: pass

sys.path.append('../../')
from dyson_solve import Dyson

from scipy.integrate import quad

from pydlr import kernel


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
ntau_pts = 2000
npts = 2000
n = np.linspace(-10000, 10000, 2*npts+1, dtype=int)
iw_array = (2*n+1)*1j*np.pi/beta

lamb=100

# setup reference data
print('getting ref data...')
compute_G_iw=True
if beta == 100 and npts == 10000:
    if os.path.isfile('G_iw_ref.txt'):
        G_iw_ref = np.loadtxt('G_iw_ref.txt', dtype=complex)
        print('\tdone with G(iω)...')
        compute_G_iw = False

if compute_G_iw:
    G_iw_ref   = eval_G_iw(iw_array, beta)
    print('saving ref data', np.savetxt('G_iw_ref.txt', G_iw_ref))
    print('\tdone with G(iω)...')

Sigma_iw_ref = G_iw_ref/4
print('\tdone with Σ(iω)...')

G0_iw_ref = eval_G0_iw(iw_array, beta)
print('\tdone with G0(iω)...')

def experiment(tol):
    print('running an experiment with tol = ', tol)
    dys = Dyson(lamb=lamb, eps=tol, options=dict(maxiter=5000, disp=True) )
    print(dys)

    print('gettting G(τ)+η')
    tau_l = dys.get_tau(beta)
    G_tau = eval_G_tau(tau_l, beta)
    # add random noise between η*(-1, 1)
    G_tau += tol*(2*np.random.rand(*G_tau.shape)-1)

    # Giw from Gtau via DLR
    print('computing G(τ) -> DLR -> G(iω)...')
    g_xaa = dys.dlr_from_tau(G_tau.reshape(-1,1,1))

    G_iw  = dys.eval_dlr_iom(g_xaa, iw_array, beta)
    G_iw  = G_iw.flatten()
    print('done...')

    # G0_iw from G0_tau via DLR
    print('done...\ncomputing G0(τ) -> DLR -> G0(iω)...')
    G0_tau = eval_G0_tau(tau_l, beta)
    print('done...')

    # Sigma from Dyson equation Σ = G0^-1 - G^-1
    print('Σ from dyson equation')
    Sigma_iw = (1.0/G0_iw_ref - 1.0/G_iw)
    print('done...')

    print('Σ from residual minimization')
    Sigma_moments = np.array([0.0, +0.25],dtype=complex).reshape(-1,1,1)

    result = dys.solve(G0_tau=G0_tau.reshape(-1,1,1), 
                       G_tau=G_tau.reshape(-1,1,1), 
                       Sigma_moments=Sigma_moments,
                       beta=beta, 
                       om_mesh=iw_array, 
                       )
    Sigma_iw_res = result.Sigma_iw
    Sigma_iw_res = Sigma_iw_res.flatten()
    print('done...')

    return {'dyson' : Sigma_iw, 'res' : Sigma_iw_res }

colors = ['tab:red', 'tab:blue', 'tab:green']

def add_results_to_plot(ax, results):
    assert len(ax) == 2
    c=0
    for key in results.keys():
        color=colors[c]
        dys = results[key]['dyson']
        res = results[key]['res']

        if key == 1e-4:
            ax[0].plot(n, Sigma_iw_ref.imag, 'o', ms=3, mfc='none', color='tab:red', label='ref')
            ax[0].plot(n, dys.imag, '.', ms=2, color='tab:blue', label=r'$G_{0}^{-1}-G^{-1}$ '+r'($\eta=$ '+'{:1.0e})'.format(key))
        ax[1].loglog(n, np.abs(dys-Sigma_iw_ref), ls='-', color=color, label=r'$\eta=$ '+'{:1.0e}'.format(key))
        ax[1].loglog(n, np.abs(res-Sigma_iw_ref), ls='--', color=color)
        ax[1].set_ylabel(r'$\Sigma$ asbolute error')
        c+=1

tols = [1e-4, 1e-8, 1e-12]
results = {tol : experiment(tol) for tol in tols }
fig, ax = plt.subplots(2,1,figsize=(5, 7))
ax[0].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
ax[0].set_ylim(-0.1, 0.1)
add_results_to_plot(ax, results)
ax[0].legend(frameon=True, framealpha=0.8, facecolor='white', edgecolor='none', loc='best', fontsize=8)
#ax[1].loglog(n[11000:-50], 1e-7*(((2*n+1)*np.pi/beta))[11000:-50], ls='dotted', lw=2, color='k')
ax[1].axhline(100, color='k', ls='-', label=r'$G_{0}^{-1}-G^{-1}$')
ax[1].axhline(100, color='k', ls='--', label='res. min')
ax[1].axhline(100, color='k', ls='dotted', label=r'$\mathcal{O}(\omega_{n})$')
ax[1].set_ylim(1e-16, 1e0)
ax[1].legend(frameon=True, framealpha=0.8, facecolor='white', edgecolor='none', ncols=2, loc='lower left', fontsize=8)
ax[1].set_xlabel(r'$n$')
plt.show(); #plt.savefig('dyson_exact_case_abserr.pdf')
