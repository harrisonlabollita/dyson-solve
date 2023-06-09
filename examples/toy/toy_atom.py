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

from triqs.gf import *

def compute_G0_iom(beta, U, n_iw=1666):
    iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=n_iw)
    G = Gf(mesh=iw_mesh, target_shape=[1,1])
    G << inverse(iOmega_n + U/2)
    return G

def compute_G0_tau(beta, U, n_tau=10000):
    tau_mesh=MeshImTime(beta=beta, S='Fermion', n_tau=n_tau)
    G = Gf(mesh=tau_mesh, target_shape=[1,1])
    eps = -0.5*U
    for tau in G.mesh:
        G[tau] = -np.exp((beta*(eps<0) - tau.value) * eps) / (1. + np.exp(-beta * abs(eps)))
    return G

def compute_G_iom(beta, U, n_iw=1666):
    iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=n_iw)
    G = Gf(mesh=iw_mesh, target_shape=[1,1])
    G << 0.5*inverse(iOmega_n + U/2) + 0.5*inverse(iOmega_n - U/2)
    return G

def compute_G_tau(beta, U, n_tau=10000):
    tau_mesh=MeshImTime(beta=beta, S='Fermion', n_tau=n_tau)
    G = Gf(mesh=tau_mesh, target_shape=[1,1])
    eps1, eps2 = 0.5*U, -0.5*U
    for tau in G.mesh:
        G[tau] = -0.5*np.exp((beta*(eps1<0) - tau.value) * eps1) / (1. +np.exp(-beta * eps1)) + \
                 -0.5*np.exp((beta*(eps2>0) - tau.value) * eps2) / (1. +np.exp(-beta * eps2))
    return G

def compute_Sigma_iom(beta, U, n_iw=1666):
    iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=n_iw)
    G = Gf(mesh=iw_mesh, target_shape=[])
    G << 0.5*U + 0.25*U*U*inverse(iOmega_n)
    return G




# set up problem
beta, U = 100, 2

lamb=100

# setup reference data
print('getting ref data...')

G_iw_ref   = compute_G_iom(beta, U)
print('\tdone with G(iω)...')

Sigma_iw_ref = compute_Sigma_iom(beta, U)
print('\tdone with Σ(iω)...')

G0_iw_ref = compute_G0_iom(beta, U)
print('\tdone with G0(iω)...')

iw_array = np.array([complex(x) for x in Sigma_iw_ref.mesh])

def experiment(tol):
    print('running an experiment with tol = ', tol)
    dys = Dyson(lamb, eps=tol, options=dict(
                                            maxiter=5000, 
											#gtol=1e-16,
                                            #xtol=1e-16,
                                            #finite_diff_rel_step=1e-20
                                            ))
    print(dys)

    print('gettting G(τ)+η')
    G_tau = compute_G_tau(beta, U)
    # add random noise between η*(-1, 1)
    G_tau.data[:] += tol*(2*np.random.rand(*G_tau.data.shape)-1)

    if False:
        tau_pts = np.array([float(x) for x in G_tau.mesh])
        g_xaa = dys.fit_dlr_from_tau(tau_pts, G_tau.data, beta)
        G_iw = G_iw_ref.copy()
        G_iw.zero()
        G_iw.data[:]  = dys.eval_dlr_iom(g_xaa, iw_array, beta)

    tail = make_zero_tail(G_iw_ref, 2); tail[1] = 1.0
    G_iw = G_iw_ref.copy()
    G_iw << Fourier(G_tau, known_moments=tail)
    
    print('done...')

    # G0_iw from G0_tau via DLR
    print('done...\ncomputing G0(τ)...')
    G0_tau = compute_G0_tau(beta, U)
    print('done...')

    # Sigma from Dyson equation Σ = G0^-1 - G^-1
    print('Σ from dyson equation')
    Sigma_iw = inverse(G0_iw_ref) - inverse(G_iw)
    print('done...')

    print('Σ from residual minimization')
    Sigma_moments = np.array([0.5*U, 0.25*U*U],dtype=complex).reshape(-1,1,1)
    
    tau_array = np.array([float(x) for x in G_tau.mesh])

    result = dys.solve(G0_tau=G0_tau.data,
                       G_tau=G_tau.data,
                       Sigma_moments=Sigma_moments,
                       beta=beta, 
                       tau_mesh=tau_array,
                       om_mesh=iw_array
                       )

    Sigma_iw_res = result.Sigma_iw
    Sigma_iw_res = Sigma_iw_res.flatten()
    print('done...')

    dyson  = Sigma_iw.data.flatten(); np.savetxt(f'dyson_{tol}.txt', dyson)
    resmin = Sigma_iw_res; np.savetxt(f'resmin_{tol}.txt', resmin)

    return {'dyson' : dyson, 'res' :  resmin}

colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']

def add_results_to_plot(ax, results):
    #assert len(ax) == 2
    c=0
    for key in results.keys():
        color=colors[c]
        dys = results[key]['dyson']
        res = results[key]['res']

        sigmax = np.max(np.abs(Sigma_iw_ref.data))

        if key == 1e-4:
            ax[0].plot(iw_array.imag, dys.real, '.', ms=2, color='tab:blue', 
                    label=r'$G_{0}^{-1}-G^{-1}$ ')
            ax[0].plot(iw_array.imag, Sigma_iw_ref.data.real, 'o', ms=2, mfc='none', color='tab:red', label=r'$\Sigma_{\mathrm{exact}}$')
            ax[0].plot(iw_array.imag, res.real, '.', ms=1, mfc='none', color='tab:green', label='res. min')
            ax[1].plot(iw_array.imag, dys.imag, '.', ms=2, color='tab:blue', 
                    label=r'$G_{0}^{-1}-G^{-1}$ ')
            ax[1].plot(iw_array.imag, Sigma_iw_ref.data.imag, 'o', ms=2, mfc='none', color='tab:red', label=r'$\Sigma_{\mathrm{exact}}$')
            ax[1].plot(iw_array.imag, res.imag, '.', ms=1, mfc='none', color='tab:green', label='res. min')

        ax[2].loglog(iw_array.imag, np.abs(dys-Sigma_iw_ref.data), ls='-', color=color, label=r'$\eta=$ '+'{:1.0e}'.format(key))
        ax[2].loglog(iw_array.imag, np.abs(res-Sigma_iw_ref.data), ls='--', color=color)
        ax[2].set_ylabel(r'$\Sigma$ absolute error')
        c+=1

def fetch_results(tols):
    results = {}

    for tol in tols:
        resmin = f'resmin_{tol}.txt'
        dyson  = f'dyson_{tol}.txt'
        if all(map(os.path.isfile, [resmin, dyson])):
            results[tol] = {'dyson' : 
                            np.loadtxt(dyson, dtype=complex), 
                            'res' : np.loadtxt(resmin, dtype=complex) }
        else: results[tol] =  experiment(tol)
    return results

tols = [1e-4, 1e-6, 1e-8]# 1e-12]
results = fetch_results(tols)

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(5, 5))
gs  = gridspec.GridSpec(2,2)

ax  = [None]*3
ax[0] = fig.add_subplot(gs[0, 0])
ax[1] = fig.add_subplot(gs[0, 1])
ax[2] = fig.add_subplot(gs[1, :])

#fig, ax = plt.subplots(2,1,figsize=(5, 7))
ax[1].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
ax[1].set_xlabel(r'$i\nu_{n}$')
ax[1].set_ylim(-0.1, 0.1)
ax[0].set_ylabel(r'Re$\Sigma(i\nu_{n})$')
ax[0].set_xlabel(r'$i\nu_{n}$')
ax[0].set_ylim(0.95, 1.05)
add_results_to_plot(ax, results)

ax[0].legend(frameon=True, framealpha=0.8, facecolor='white', edgecolor='none', loc='lower left', fontsize=8)
ax[2].loglog(iw_array.imag[1700:1900], 1e-3*(iw_array.imag**2)[1700:1900], ls='dotted', lw=2, color='k')
ax[2].axhline(100, color='k', ls='-', label=r'$G_{0}^{-1}-G^{-1}$')
ax[2].axhline(100, color='k', ls='--', label='res. min.')
ax[2].axhline(100, color='k', ls='dotted', label=r'$\mathcal{O}(\omega_{n}^{2})$')
ax[2].set_ylim(1e-14, 1e0)
ax[2].legend(frameon=True, framealpha=0.8, facecolor='white', edgecolor='none', ncols=2, loc='lower left', fontsize=7)
ax[2].set_xlabel(r'$i\nu_{n}$')

for a, let in zip(ax, ['(a)', '(b)', '(c)']):
    t = a.text(0.03, 0.85, let, transform = a.transAxes, size=14) 
    t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))

plt.subplots_adjust(wspace=0.6, hspace=0.4)
#plt.show();
plt.savefig('dyson_exact_atom_abs_err.pdf')
