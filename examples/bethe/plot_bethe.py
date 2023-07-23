#!/usr/bin/env python3
import sys, os

import numpy as np

import matplotlib.pyplot as plt
try: plt.style.use('publish') 
except: pass
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    
    ax.semilogy(lambdas, ref_cvg, 'o-', mec=color[0], color=color[1], mfc='none', )
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
    G_w.set_from_pade(G_iw['up'], n_points=1000, freq_offset=0.01)
    om = np.array([float(x) for x in w_mesh])

    ax.plot(om, -G_w.data[:,0,0].imag/np.pi, '-', color='tab:blue', lw=2)
    y = 13/48
    ax.arrow(0, y, 4, 0, color='k', head_width=0.015, head_length=0.15, length_includes_head=True)
    ax.arrow(4, y, -4, 0, color='k', head_width=0.015, head_length=0.15, length_includes_head=True)
    ax.text(3/8, y+0.01, r'$\omega_{\mathrm{max}} = 4$ eV')
    ax.text(-4.5, y+0.01, r'$\beta = 5$ eV$^{-1}$')
    ax.axvspan(0, +4, alpha=0.1, lw=0, facecolor='k')
    #ax.axvline(0.0, color='k', ls='dotted')
    #ax.axvline(4.0, color='k', ls='dotted')

    ax.set_ylim(0,y+0.2)
    ax.set_xlim(-5,5)
    ax.set_xticks([-4, -2, 0, 2, 4])
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
    
    scale = 1.2
    fig, ax = plt.subplots(2,2, figsize=(6*scale, 6*scale))

    ins = inset_axes(ax[0,1], width="35%", height="40%", loc=4, borderpad=2)

    plot_Aw(ax[0,0], datainfo['parent_dir'])
    plot_convergence(ax[1,0], 5e6, ['tab:blue', 'tab:blue'], **datainfo);
    plot_convergence(ax[1,0], 5e9, ['tab:red','tab:red'], **datainfo);

    add_label(ax[1,0], color='k', marker='o', mfc='none', lw=2, label=r'$\Sigma_{\Lambda_{i}}-\Sigma_{\mathrm{ref}}$')
    add_label(ax[1,0], color='k', marker='x', mfc='none', lw=2, label=r'$\Sigma_{\Lambda_{i}}-\Sigma_{\Lambda_{c}}$')
    add_label(ax[1,0], color='tab:blue', ls='-',  lw=2, label=r'N = $\mathcal{O}(10^{6})$')
    add_label(ax[1,0], color='tab:red', ls='-', lw=2, label=r'N = $\mathcal{O}(10^{9})$')
    ax[1,0].legend(frameon=True, framealpha=0.8, edgecolor='white', facecolor='white', fontsize=7.5, ncols=2)
    ax[1,0].set_xlim(-5,)

    parent_dir = datainfo['parent_dir']

    ar = HDFArchive(parent_dir+'/'+file_base.format(0, 10001, 1e11))
    S_ref =  ar['Sigma_iw']

    ar = HDFArchive(parent_dir+'/'+file_base.format(2, 10001, 5e9))
    S_fit =  ar['Sigma_iw_fit']

    mesh = np.array([complex(x) for x in S_fit.mesh])
    idx = np.where(mesh > 0)
    #ax[2].plot(mesh.imag, S_ref['up'][0,0].data.real-S_ref['up'](0)[0,0].real, '.', ms=2, color='tab:blue')
    #ax[2].plot(mesh.imag, S_fit['up'][0,0].data.real-S_fit['up'](0)[0,0].real, 'o', ms=3, mfc='none', color='tab:red')
    ax[0,1].plot(mesh[idx].imag, S_fit['up'][0,0].data[idx].imag, 'o',markeredgewidth=1.5, mfc='none', lw=2, color='tab:blue', label=r'$\Lambda = 2$')
    ins.plot(mesh[idx].imag, S_fit['up'][0,0].data[idx].imag, 'o', markeredgewidth=1.5, mfc='none', lw=2, color='tab:blue', label=r'$\Lambda = 2$')
    ax[0,1].set_xlabel(r'$\nu_{n}$'); ax[0,1].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
    ax[0,1].set_xlim(0, 20); ax[0,1].set_ylim(-1.1, 0) #-0.25)
    ax[1,1].loglog(mesh.imag, np.abs(S_ref['up'][0,0].data-S_fit['up'][0,0].data), '-', lw=2, color='tab:blue', label=r'$\Lambda = 2$')

    #ar = HDFArchive(parent_dir+'/'+file_base.format(20, 10001, 5e6))
    #S_fit =  ar['Sigma_iw_fit']
    #ax[3].loglog(mesh.imag, np.abs(S_ref['up'][0,0].data-S_fit['up'][0,0].data), '-', lw=2, color='tab:red')
    #ax[3].loglog(mesh.imag, S_ref['up'][0,0].data.imag, '.', ms=2, color='tab:blue')
    ax[1,1].set_xlabel(r'$\nu_{n}$'); #ax.set_ylabel(r'$\Sigma(i\nu_{n})$')

    ar = HDFArchive(parent_dir+'/'+file_base.format(6, 10001, 5e9))
    S_fit =  ar['Sigma_iw_fit']
    mesh = np.array([complex(x) for x in S_fit.mesh])
    idx = np.where(mesh > 0)
    ax[0,1].plot(mesh.imag[idx], S_fit['up'][0,0].data[idx].imag, 'x', markeredgewidth=1.5, lw=2, color='tab:red', label=r'$\Lambda = 6$')
    ins.plot(mesh.imag[idx], S_fit['up'][0,0].data[idx].imag, 'x', markeredgewidth=1.5, lw=2, color='tab:red', label=r'$\Lambda = 6$')
    ax[1,1].loglog(mesh.imag, np.abs(S_ref['up'][0,0].data-S_fit['up'][0,0].data), '-', lw=2, color='tab:red', label=r'$\Lambda = 6$')

    ar = HDFArchive(parent_dir+'/'+file_base.format(20, 10001, 5e9))
    S_fit =  ar['Sigma_iw_fit']
    mesh = np.array([complex(x) for x in S_fit.mesh])
    idx = np.where(mesh > 0)
    ax[0,1].plot(mesh.imag[idx], S_fit['up'][0,0].data[idx].imag, '^',markeredgewidth=1.5, mfc='none', lw=2, color='tab:green', label=r'$\Lambda = 20$')
    ins.plot(mesh.imag[idx], S_fit['up'][0,0].data[idx].imag, '^', markeredgewidth=1.5, mfc='none', lw=2, color='tab:green', label=r'$\Lambda = 20$')
    ax[1,1].loglog(mesh.imag, np.abs(S_ref['up'][0,0].data-S_fit['up'][0,0].data), '-', lw=2, color='tab:green', label=r'$\Lambda = 20$')
    ax[1,1].legend(loc='best')
    ax[1,1].set_ylabel(r'$|\Sigma_{\Lambda}(i\nu_{n})-\Sigma_{\mathrm{ref}}(i\nu_{n})|$')

    ax[0,1].plot(mesh[idx].imag, S_ref['up'][0,0].data[idx].imag, 'o', lw=2, mfc='none', mec='k', label='ref')
    ins.plot(mesh[idx].imag, S_ref['up'][0,0].data[idx].imag, 'o', lw=2, mfc='none', mec='k', label='ref')
    ax[0,1].legend(loc='best')
    ins.set_xlim(3, 7); ins.set_ylim(-0.9, -0.55)
    ins.tick_params(labelsize=9)
    #plot_sigma_cmp(ax[2], datainfo['parent_dir'])
    #plt.subplots_adjust(wspace=0.4, hspace=1/3)
    #fig, ax = plt.subplots(2, 1, figsize=(4,6), sharex=True)
    #plot_example(ax, 5e9, 'tab:blue', n_tau=10001, lambdas=20, parent_dir='data_beta_5')

    for a, let, coords in zip(ax.flatten(), ['(a)', '(b)', '(c)', '(d)'], 
                             [(0.05, 0.90), (0.87, 0.90), (0.05, 0.10), (0.87, 0.10)]):
        t = a.text(coords[0], coords[1], let, transform = a.transAxes, size=14) 
        t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))

    plt.subplots_adjust(hspace=0.25, wspace=0.37)
    plt.show()
    #plt.savefig('int_bethe_problem.pdf')
