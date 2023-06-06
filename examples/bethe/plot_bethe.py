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
    
    ax.semilogy(lambdas, ref_cvg, 'o-', mec=color[0], color=color[1])
    ax.semilogy(lambdas[:-1], np.abs((ref_cvg-ref_cvg[-1]))[:-1], 'x-', mec=color[0], color=color[1])
    ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$L^{\infty}$ error of $\Sigma(i\nu_{n})$')
    ax.axvline(20, color='k', ls='dotted', lw=1)
    ax.arrow(0.52, 0.19, -0.1, 0, color='k', head_width=0.02, head_length=0.01, 
            length_includes_head=True, transform=ax.transAxes)
    ax.text(0.53, 0.175, r"$\Lambda \sim \beta\omega_{\mathrm{max}}$", transform=ax.transAxes)

def plot_Aw(ax, parent_dir):
    ar = HDFArchive(parent_dir+'/'+file_base.format(0, 10001, 1e11))
    G_iw = ar['G_iw']

    w_mesh = MeshReFreq(window=(-8,8), n_w=5000)
    G_w = Gf(mesh=w_mesh, target_shape=[1,1])
    G_w.set_from_pade(G_iw['up'])
    om = np.array([float(x) for x in w_mesh])
    ax.plot(om, -G_w.data[:,0,0].imag/np.pi, lw=2)
    y = 13/24
    ax.arrow(0, y, 4, 0, color='k', head_width=0.015, head_length=0.15, length_includes_head=True)
    ax.arrow(4, y, -4, 0, color='k', head_width=0.015, head_length=0.15, length_includes_head=True)
    ax.text(3/8, y+0.01, r'$\omega_{\mathrm{max}} = 4$ eV')
    ax.text(-5, y+0.01, r'$\beta = 5$ eV$^{-1}$')
    ax.axvspan(0, +4, color='k', alpha=0.1, lw=0)
    #ax.axvline(0.0, color='k', ls='dotted')
    #ax.axvline(4.0, color='k', ls='dotted')

    ax.set_ylim(0,y+0.2)
    ax.set_xlim(-6,6)
    ax.set_xlabel(r'$\omega$'); ax.set_ylabel(r'$-\mathrm{Im}G(\omega)/\pi$')

def plot_sigma_cmp(ax, parent_dir):
    ar = HDFArchive(parent_dir+'/'+file_base.format(0, 10001, 1e11))
    S_ref =  ar['Sigma_iw']
    ar = HDFArchive(parent_dir+'/'+file_base.format(20, 10001, 5e9))
    S_fit =  ar['Sigma_iw_fit']

    mesh = np.array([complex(x) for x in S_fit.mesh])

    ax.plot(mesh.imag, S_fit['up'][0,0].data.real-S_fit['up'](0)[0,0].real, 'o', ms=3, mfc='none')
    ax.plot(mesh.imag, S_fit['up'][0,0].data.imag, 'o', ms=3, mfc='none')
    ax.plot(mesh.imag, S_ref['up'][0,0].data.real-S_ref['up'](0)[0,0].real, '.', ms=2)
    ax.plot(mesh.imag, S_ref['up'][0,0].data.imag, '.', ms=2)
    ax.set_xlabel(r'$i\nu_{n}$'); ax.set_ylabel(r'$\Sigma(i\nu_{n})$')



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
    ax[1].semilogy(iw.imag, np.abs(S_ref['up'].data.flatten()-S_fit['up'].data.flatten()))#/sig_max)
    #ax[1].set_ylim(1e-6, 1e-2)
   #ax[1].set_xlim(0,)
    

add_label = lambda ax, **kwargs : ax.axvline(-100,**kwargs)

if __name__ == "__main__":

    datainfo =  {
            'parent_dir' : 'data_beta_5',
            'lambdas'   :  [1, 2, 4, 6, 8, 10, 12, 15, 20, 22, 25, 27, 30, 40, 50],
            'mc_cycles' :  [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9],
            'n_taus'    :  [10001]
           }

    fig, ax = plt.subplots(2, 1, figsize=(3,6))


    plot_Aw(ax[0], datainfo['parent_dir'])
    plot_convergence(ax[1], 5e6, ['tab:red', 'lightcoral'], **datainfo);
    plot_convergence(ax[1], 5e9, ['tab:blue', 'deepskyblue'], **datainfo);

    add_label(ax[1], color='k', marker='o', lw=2, label=r'$\Sigma_{\Lambda_{i}}-\Sigma_{\mathrm{ref}}$')
    add_label(ax[1], color='k', marker='x', mfc='none', lw=2, label=r'$\Sigma_{\Lambda_{i}}-\Sigma_{\Lambda_{c}}$')
    add_label(ax[1], color='lightcoral', ls='-',  lw=2, label=r'N = $\mathcal{O}(10^{6})$')
    add_label(ax[1], color='deepskyblue', ls='-', lw=2, label=r'N = $\mathcal{O}(10^{9})$')
    ax[1].legend(frameon=True, framealpha=0.8, edgecolor='white', facecolor='white', fontsize=7.5, ncols=2)

    ax[1].set_xlim(-5,)

    #plot_sigma_cmp(ax[2], datainfo['parent_dir'])
    plt.subplots_adjust(wspace=0.4, hspace=1/3)
    #fig, ax = plt.subplots(2, 1, figsize=(4,6), sharex=True)
    #plot_example(ax, 5e9, 'tab:blue', n_tau=10001, lambdas=20, parent_dir='data_beta_5')

    for a, let in zip(ax, ['(a)', '(b)']):
        #t = a.text(0.03, 0.85, let, transform = a.transAxes, size=14) 
        t = a.text(0.03, 0.15, let, transform = a.transAxes, size=14) 
        t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))

    plt.savefig('int_bethe_problem.pdf')
    #plt.show()
