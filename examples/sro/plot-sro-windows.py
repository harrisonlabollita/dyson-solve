import sys, glob, os
from h5 import HDFArchive
from triqs.gf import *
import scipy.integrate

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('publish')
import warnings
from matplotlib.gridspec import GridSpec
warnings.simplefilter(action='ignore', category=FutureWarning)

def add_convergence_to_plot(ax, data, **kwargs):
    mu, nlatt, nimp, Gimp = [], [], [], []
    maxiter = min(data['dmft_results']['iterations'], 30)
    for it in range(maxiter):
        mu.append(data[f'dmft_results/it_{it}']['mu'])
        nlatt.append(data[f'dmft_results/it_{it}']['n_latt'])
        nimp.append(data[f'dmft_results/it_{it}']['n_imp'])
        Gimp.append(data[f'dmft_results/it_{it}']['Gloc'])
    mu, nimp, nlatt = np.array(mu), np.array(nimp), np.array(nlatt)
    ax[0].semilogy(list(range(1, maxiter)), np.abs(mu[:-1]-mu[-1]), 'o-', **kwargs); 
    #ax[0].semilogy(list(range(1, maxiter)), np.abs(np.diff(mu)), 'o-', **kwargs); 
    ax[0].set_ylabel(r'$\delta\mu$')
    #ax[1].semilogy(range(maxiter), np.abs(nimp-nlatt), 'o-', **kwargs); 
    #ax[1].semilogy(list(range(1,maxiter)), np.abs(np.diff(nimp)), 'o-', **kwargs); 
    #ax[1].semilogy(list(range(1,maxiter)), np.abs(nimp[:-1]-nlatt[-1]), 'o-', **kwargs); 
    #ax[1].set_ylabel(r'$\delta n$')

    ax[1].semilogy(list(range(1,maxiter)), [np.max(np.abs(Gimp[i]['up'].data-Gimp[-1]['up'].data)) for i in range(len(Gimp)-1)], 
                   'o-', **kwargs); 
    ax[1].set_ylabel(r'$\delta G$')
    ax[1].set_ylim(1e-3,)
    ax[0].set_xticklabels([])
    ax[0].legend(loc='upper right', fontsize=8)

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

    orb, name = 0, 'dxz'
    Sigma_high00 = Sigma_moments['up'][0][orb,orb]+Sigma_moments['up'][1][orb,orb]/iw

    #fig, ax = plt.subplots(1,2,figsize=(4,5), width_ratios=[2,1])
    scale = 1.2
    fig,ax = plt.subplots(2,3,width_ratios=[2,1,2], figsize=(9*scale, 4*scale))

    higher = np.where(iw.imag > 5)

    windows = { 'window1' : (2,4),
                'window2' : (4,6),
                'window3' : (6,10)
              }

    #ax[1,0].axvspan(*windows['window3'], color='tab:red',    alpha=0.25, lw=1)
    #ax[1,0].axvspan(*windows['window2'], color='purple',       alpha=0.25, lw=1)
    #ax[1,0].axvspan(*windows['window1'], color='dodgerblue', alpha=0.25, lw=1)

    #ax[0].text(2.2, -0.1, 'window 1', fontsize=9, color='dodgerblue', rotation=65)
    #ax[0].text(4.2, -0.2, 'window 2', fontsize=9, color='cyan', rotation=65)
    #ax[0].axvline(2, color='dodgerblue', ls='-', lw=1)
    #ax[0].axvline(2, color='cyan', ls='--', lw=1)
    #ax[0].axvline(2, color='tab:red', ls='-.', lw=1)
    #ax[0].axvline(4, color='dodgerblue', ls='-', lw=1)
    #ax[0].axvline(6, color='cyan', ls='--', lw=1)
    #ax[0].axvline(10, color='tab:red', ls='-.', lw=1)
    ax[0,0].plot(iw.imag, Siw_raw['up'].data[:,orb,orb].real, 'ko',ms=4,mfc='none', label=r'$G_{0}^{-1}-G^{-1}$')
    ax[0,1].plot(iw.imag, Siw_raw['up'].data[:,orb,orb].real, 'ko',ms=4,mfc='none', label=r'$G_{0}^{-1}-G^{-1}$')
    ax[1,0].plot(iw.imag, Siw_raw['up'].data[:,orb,orb].imag, 'ko',ms=4,mfc='none', label=r'$G_{0}^{-1}-G^{-1}$')
    ax[1,1].plot(iw.imag, Siw_raw['up'].data[:,orb,orb].imag, 'ko',ms=4,mfc='none', label=r'$G_{0}^{-1}-G^{-1}$')

    ax[0,0].plot(iw.imag, Siw_tailsm['up'].data[:,orb,orb].real, '-', color='tab:blue', label='window 1')
    ax[0,1].plot(iw.imag, Siw_tailsm['up'].data[:,orb,orb].real, '-', color='tab:blue', label='window 1')
    ax[1,0].plot(iw.imag, Siw_tailsm['up'].data[:,orb,orb].imag, '-', color='tab:blue', label='window 1')
    ax[1,1].plot(iw.imag, Siw_tailsm['up'].data[:,orb,orb].imag, '-', color='tab:blue', label='window 1')

    ax[0,0].plot(iw.imag, Siw_tailmid['up'].data[:,orb,orb].real, '-',color='tab:red', label='window 2')
    ax[0,1].plot(iw.imag, Siw_tailmid['up'].data[:,orb,orb].real, '-',color='tab:red', label='window 2')
    ax[1,0].plot(iw.imag, Siw_tailmid['up'].data[:,orb,orb].imag, '-',color='tab:red', label='window 2')
    ax[1,1].plot(iw.imag, Siw_tailmid['up'].data[:,orb,orb].imag, '-',color='tab:red', label='window 2')

    ax[0,0].plot(iw.imag, Siw_taillg['up'].data[:,orb,orb].real, '-', color='tab:green', label='window 3')
    ax[0,1].plot(iw.imag, Siw_taillg['up'].data[:,orb,orb].real, '-', color='tab:green', label='window 3')
    ax[1,0].plot(iw.imag, Siw_taillg['up'].data[:,orb,orb].imag, '-', color='tab:green', label='window 3')
    ax[1,1].plot(iw.imag, Siw_taillg['up'].data[:,orb,orb].imag, '-', color='tab:green', label='window 3')

    ax[0,0].plot(iw.imag, Siw_res['up'].data[:,orb,orb].real, '-', color='tab:orange', label='res. min.')
    ax[0,1].plot(iw.imag, Siw_res['up'].data[:,orb,orb].real, '-', color='tab:orange', label='res. min.')
    ax[1,0].plot(iw.imag, Siw_res['up'].data[:,orb,orb].imag, '-', color='tab:orange', label='res. min.')
    ax[1,1].plot(iw.imag, Siw_res['up'].data[:,orb,orb].imag, '-', color='tab:orange', label='res. min.')

    ax[0,0].plot(iw.imag[higher], Sigma_high00[higher].real, '--', color='tab:purple', alpha=0.9, label=r'$\Sigma_{\infty}+\Sigma_{1}/i\nu_{n}$')
    ax[0,1].plot(iw.imag[higher], Sigma_high00[higher].real, '--', color='tab:purple', alpha=0.9, label=r'$\Sigma_{\infty}+\Sigma_{1}/i\nu_{n}$')
    ax[1,0].plot(iw.imag[higher], Sigma_high00[higher].imag, '--', color='tab:purple', alpha=0.9, label=r'$\Sigma_{\infty}+\Sigma_{1}/i\nu_{n}$')
    ax[1,1].plot(iw.imag[higher], Sigma_high00[higher].imag, '--', color='tab:purple', alpha=0.9, label=r'$\Sigma_{\infty}+\Sigma_{1}/i\nu_{n}$')


    ax[1,0].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
    ax[0,0].set_ylabel(r'Re$\Sigma(i\nu_{n})$')
    ax[1,0].set_xlabel(r'$\nu_{n}$')
    ax[1,1].set_xlabel(r'$\nu_{n}$')
    ax[0,0].set_xlim(0,12); ax[0,0].set_ylim(4.5, 5.5); 
    ax[1,0].set_xlim(0,12); ax[1,0].set_ylim(-0.5, -0.1); 
    ax[0,0].legend(frameon=True, framealpha=0.5, facecolor='white', edgecolor='none', loc='lower left', ncols=2, fontsize=9)
    ax[0,1].set_xlim(60,120)
    ax[0,1].set_ylim(4.85, 5.1)
    ax[0,1].set_ylabel(r'Re$\Sigma(i\nu_{n})$')

    ax[1,1].set_xlim(60,120)
    ax[1,1].set_ylim(-0.07, 0); ax[1,1].set_yticks([-0.06, -0.03, 0.0])
    ax[1,1].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
    ax[1,1].set_xlabel(r'$\nu_{n}$')

    add_convergence_to_plot(ax[:,-1], HDFArchive('sro_dmft_tail_sm_wind.h5'), label='window1', color='tab:blue')
    add_convergence_to_plot(ax[:,-1], HDFArchive('sro_tailfit_mc_1e8.h5'), label='window2', color='tab:red')
    add_convergence_to_plot(ax[:,-1], HDFArchive('sro_dmft_tail_lg_wind.h5'), label='window3', color='tab:green')
    add_convergence_to_plot(ax[:,-1], res, label='res. min.', color='tab:orange')
    
    #ax[1].plot(iw.imag, Siw_raw['up'].data[:,orb,orb].imag, 'o',ms=3, label='Dyson')
    #ax[1].plot(iw.imag, Siw_tailsm['up'].data[:,orb,orb].imag, '-',label='window1')
    #ax[1].plot(iw.imag, Siw_tailmid['up'].data[:,orb,orb].imag, '-',label='window2')
    #ax[1].plot(iw.imag, Siw_taillg['up'].data[:,orb,orb].imag, '-',label='window3')
    #ax[1].plot(iw.imag, Siw_res['up'].data[:,orb,orb].imag, '-', label='res. min.')
    #ax[1].plot(iw.imag[higher], Sigma_high00[higher].imag, '--', alpha=0.75)
    #ax[1].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
    #ax[1].set_xlabel(r'$\nu_{n}$')
    ax[0,0].set_xticklabels([]); ax[0,1].set_xticklabels([])

    plt.subplots_adjust(hspace=0.1, wspace=0.5)

    for a, let in zip([ax[0,0], ax[1,0]], 
                      ['(a)', '(b)']): 
        t = a.text(0.05, 0.85, let, transform = a.transAxes, size=14) 
                                  #backgroundcolor='white', alpha=0.75)
        t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))

    for a, let in zip([ax[0,1], ax[1,1], ax[0,2], ax[1,2]],['(c)', '(d)', '(e)', '(f)']):
        t = a.text(0.05, 0.15, let, transform = a.transAxes, size=14) 
                                  #backgroundcolor='white', alpha=0.75)
        t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))

    #for a, let in zip(ax, 
    #                  ['(a)', '(b)']): 
    #plt.savefig(f'sro-{name}.pdf')
    #plt.show()
    plt.savefig('sro_results.pdf')


