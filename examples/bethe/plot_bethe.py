#!/usr/bin/env python3
import sys, os

import numpy as np

import matplotlib.pyplot as plt
try: plt.style.use('publish') 
except: pass

from triqs.gf import *
from h5 import HDFArchive

file_base = 'bethe_{}_{}_{:1.0e}.h5'

def plot_convergence(ax, mc, color, n_taus, mc_cycles, lambdas, parent_dir):
    
    n_tau = n_taus[0]

    ar = HDFArchive(parent_dir+'/'+file_base.format(0, 10001, 1e11))
    S_ref =  ar['Sigma_iw']

    sig_max = max(np.abs(S_ref['up'].data.flatten()))
    del ar

    ref_cvg = [None]*len(lambdas)
    j = 0
    for lamb in lambdas:
        ar = HDFArchive(parent_dir+'/'+file_base.format(lamb, n_tau, mc))
        S_fit =  ar['Sigma_iw_fit']
        del ar
        ref_cvg[j] = np.max(np.abs(S_ref['up'].data.flatten()-S_fit['up'].data.flatten()))/sig_max
        j+=1
    
    ax.semilogy(lambdas, ref_cvg, 'o-', mfc='none', color=color)
    ax.semilogy(lambdas[:-1], np.abs((ref_cvg-ref_cvg[-1]))[:-1], 's-', mfc='none',  color=color)
    ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$L^{\infty}$ error of $\Sigma(i\nu_{n})$')

def plot_example(ax, mc, color, n_tau=10001, lambdas=20, parent_dir='data_beta_5'):
    ar = HDFArchive(parent_dir+'/'+file_base.format(0, 10001, 1e11))
    S_ref =  ar['Sigma_iw']
    sig_max = max(np.abs(S_ref['up'].data.flatten()))

    ar = HDFArchive(parent_dir+'/'+file_base.format(lambdas, n_tau, mc))
    S_fit =  ar['Sigma_iw_fit']

    iw = np.array([complex(x) for x in S_ref.mesh])
    ax[0].plot(iw.imag, S_ref['up'].data.flatten().real-S_ref['up'](0)[0,0].real, 'o', ms=3) 
    ax[0].plot(iw.imag, S_ref['up'].data.flatten().imag, 'o', ms=3) 
    ax[0].plot(iw.imag, S_fit['up'].data.flatten().real-S_ref['up'](0)[0,0].real, '.', ms=2) 
    ax[0].plot(iw.imag, S_fit['up'].data.flatten().imag, '.', ms=2) 
    ax[1].semilogy(iw.imag, np.abs(S_ref['up'].data.flatten()-S_fit['up'].data.flatten())/sig_max)
    ax[1].set_ylim(1e-6, 1e-2)
    ax[1].set_xlim(0,)
    

add_label = lambda ax, **kwargs : ax.axvline(-100,**kwargs)

if __name__ == "__main__":

    datainfo =  {
            'parent_dir' : 'data_beta_5',
            'lambdas'   :  [1, 2, 4, 6, 8, 10, 12, 15, 20, 22, 25, 27, 30, 40, 50],
            'mc_cycles' :  [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9],
            'n_taus'    :  [10001]
           }

    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    plot_convergence(ax, 5e8, 'tab:red', **datainfo);
    plot_convergence(ax, 5e9, 'tab:blue', **datainfo);
    add_label(ax, color='k', mfc='none', marker='o', lw=2, label=r'$\Sigma_{\Lambda_{i}}-\Sigma_{\mathrm{ref}}$')
    add_label(ax,mfc='none', color='k', marker='s', lw=2, label=r'$\Sigma_{\Lambda_{i}}-\Sigma_{\Lambda_{c}}$')
    add_label(ax, color='tab:red', ls='-',  lw=2, label=r'N = $\mathcal{O}(10^{8})$')
    add_label(ax, color='tab:blue', ls='-', lw=2, label=r'N = $\mathcal{O}(10^{9})$')
    ax.legend()
    ax.set_xlim(0, 60)
    fig, ax = plt.subplots(2, 1, figsize=(4,6), sharex=True)
    plot_example(ax, 5e9, 'tab:blue', n_tau=10001, lambdas=20, parent_dir='data_beta_5')

    #plt.savefig('lambda_cvg.pdf')
    plt.show()
