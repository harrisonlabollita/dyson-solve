import sys, glob, os
from h5 import HDFArchive
from triqs.gf import *
import scipy.integrate

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('publish')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def add_convergence_to_plot(ax, data, **kwargs):
    mu, nlatt, nimp= [], [], []
    maxiter = min(data['dmft_results']['iterations'], 30)
    for it in range(maxiter):
        mu.append(data[f'dmft_results/it_{it}']['mu'])
        nlatt.append(data[f'dmft_results/it_{it}']['n_latt'])
        nimp.append(data[f'dmft_results/it_{it}']['n_imp'])
    mu, nimp, nlatt = np.array(mu), np.array(nimp), np.array(nlatt)
    #ax[0].semilogy(list(range(1, maxiter)), np.abs(mu[:-1]-mu[-1]), 'o-', **kwargs); 
    ax[0].semilogy(list(range(1, maxiter)), np.abs(np.diff(mu)), 'o-', **kwargs); 
    ax[0].set_ylabel(r'$\delta\mu$')
    #ax[1].semilogy(range(maxiter), np.abs(nimp-nlatt), 'o-', **kwargs); 
    ax[1].semilogy(list(range(1,maxiter)), np.abs(np.diff(nimp)), 'o-', **kwargs); 
    ax[1].set_ylabel(r'$\delta n_{\mathrm{imp}}$')
    ax[-1].set_xlabel('Iteration')

if __name__ == "__main__":

    filename = 'sro_200_1e-06_tau_10001_mc_1e8.h5'
    res = HDFArchive(filename)
    Siw_res = res['dmft_results/last_iter']['Sigma_iw']

    Sigma_moments = res['dmft_results/last_iter']['Sigma_moments']

    Siw_raw = res['dmft_results/last_iter']['Sigma_iw_raw']
    tailsm = HDFArchive('sro_dmft_tail_sm_wind.h5')
    tailmid = HDFArchive('sro_tailfit_mc_1e8.h5')
    taillg = HDFArchive('sro_dmft_tail_lg_wind.h5')

    Siw_tailsm = tailsm['dmft_results/last_iter']['Sigma_iw']
    Siw_tailmid = tailmid['dmft_results/last_iter']['Sigma_iw']
    Siw_taillg = taillg['dmft_results/last_iter']['Sigma_iw']

    iw = np.array([complex(x) for x in Siw_res.mesh])

    for orb, name in zip([0, 2], ['dxz', 'dxy']):
        Sigma_high00 = Sigma_moments['up'][0][orb,orb]+Sigma_moments['up'][1][orb,orb]/iw

        fig, ax = plt.subplots(1,2,figsize=(4,5), width_ratios=[2,1])

        higher = np.where(iw.imag > 5)

        ax[0].plot(iw.imag, Siw_raw['up'].data[:,orb,orb].imag, 'o',ms=3, label=r'$G_{0}^{-1}-G^{-1}$')
        ax[0].plot(iw.imag, Siw_res['up'].data[:,orb,orb].imag, '-', label='res. min.')
        ax[0].plot(iw.imag, Siw_tailsm['up'].data[:,orb,orb].imag, '-',label='window1')
        ax[0].plot(iw.imag, Siw_tailmid['up'].data[:,orb,orb].imag, '-',label='window2')
        ax[0].plot(iw.imag, Siw_taillg['up'].data[:,orb,orb].imag, '-',label='window3')
        ax[0].plot(iw.imag[higher], Sigma_high00[higher].imag, '--', alpha=0.75)
        ax[0].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
        ax[0].set_xlabel(r'$\nu_{n}$')
        ax[0].set_xlim(0,20)
        ax[0].set_ylim(-0.5, 0.); 
        ax[0].legend(frameon=True, framealpha=0.5, facecolor='white', edgecolor='none', loc='lower right')
    
        ax[1].plot(iw.imag, Siw_raw['up'].data[:,orb,orb].imag, 'o',ms=3, label='Dyson')
        ax[1].plot(iw.imag, Siw_res['up'].data[:,orb,orb].imag, '-', label='res. min.')
        ax[1].plot(iw.imag, Siw_tailsm['up'].data[:,orb,orb].imag, '-',label='window1')
        ax[1].plot(iw.imag, Siw_tailmid['up'].data[:,orb,orb].imag, '-',label='window2')
        ax[1].plot(iw.imag, Siw_taillg['up'].data[:,orb,orb].imag, '-',label='window3')
        ax[1].plot(iw.imag[higher], Sigma_high00[higher].imag, '--', alpha=0.75)
        ax[1].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
        ax[1].set_xlabel(r'$\nu_{n}$')
        ax[1].set_xlim(60,120)
        ax[1].set_ylim(-0.035, -0.015)

        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        #for a, let in zip([ax[0,0], ax[0,1], ax[1,0], ax[1,1]], 
        #                  ['(a)', '(b)', '(c)', '(d)']): 
        for a, let in zip(ax, 
                          ['(a)', '(b)']): 
            t = a.text(0.03, 0.85, let, transform = a.transAxes, size=14) 
                                      #backgroundcolor='white', alpha=0.75)
            #t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))
        #plt.savefig(f'sro-{name}.pdf')


    fig, ax = plt.subplots(2,1,sharex=True, figsize=(5,6))
    add_convergence_to_plot(ax, HDFArchive('sro_dmft_tail_sm_wind.h5'), label='window1', color='tab:green', mfc='none')
    add_convergence_to_plot(ax, HDFArchive('sro_tailfit_mc_1e8.h5'), label='window2', color='tab:blue', mfc='none')
    add_convergence_to_plot(ax, HDFArchive('sro_dmft_tail_lg_wind.h5'), label='window3', color='tab:red', mfc='none')
    add_convergence_to_plot(ax, res, label='res. min.', color='tab:orange', mfc='none')
    ax[0].legend()
    for a, let in zip([ax[0], ax[1]], 
                      ['(a)', '(b)']): 
        t = a.text(0.03, 0.05, let, transform = a.transAxes, size=14) 
        t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))

    plt.show()#plt.savefig('sro-convergence.pdf')
