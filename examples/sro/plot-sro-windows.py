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
    fig,ax = plt.subplots(1,2,width_ratios=[2,1], figsize=(7, 3))


    higher = np.where(iw.imag > 5)

    windows = { 'window1' : (2,4),
                'window2' : (2,6),
                'window3' : (2,10)
              }
    ax[0].plot(iw.imag, Siw_raw['up'].data[:,orb,orb].imag, 'o',ms=3, label=r'$G_{0}^{-1}-G^{-1}$')
    ax[1].plot(iw.imag, Siw_raw['up'].data[:,orb,orb].imag, 'o',ms=3, label=r'$G_{0}^{-1}-G^{-1}$')

    ax[0].plot(iw.imag, Siw_tailsm['up'].data[:,orb,orb].imag, '-', color='dodgerblue', label='window1')
    ax[1].plot(iw.imag, Siw_tailsm['up'].data[:,orb,orb].imag, '-', color='dodgerblue', label='window1')

    ax[0].plot(iw.imag, Siw_tailmid['up'].data[:,orb,orb].imag, '-',color='cyan', label='window2')
    ax[1].plot(iw.imag, Siw_tailmid['up'].data[:,orb,orb].imag, '-',color='cyan', label='window2')
    ax[0].plot(iw.imag, Siw_taillg['up'].data[:,orb,orb].imag, '-', color='tab:red', label='window3')
    ax[1].plot(iw.imag, Siw_taillg['up'].data[:,orb,orb].imag, '-', color='tab:red', label='window3')

    ax[0].plot(iw.imag, Siw_res['up'].data[:,orb,orb].imag, '-', color='tab:green', label='res. min.')
    ax[1].plot(iw.imag, Siw_res['up'].data[:,orb,orb].imag, '-', color='tab:green', label='res. min.')

    ax[0].axvspan(*windows['window3'], facecolor='tab:red',    alpha=0.25, lw=0,)
    ax[0].axvspan(*windows['window2'], facecolor='cyan',       alpha=0.25, lw=0,)
    ax[0].axvspan(*windows['window1'], facecolor='dodgerblue', alpha=0.25, lw=0,)

    ax[0].plot(iw.imag[higher], Sigma_high00[higher].imag, '--', color='tab:orange')
    ax[1].plot(iw.imag[higher], Sigma_high00[higher].imag, '--', color='tab:orange')

    ax[0].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
    ax[0].set_xlabel(r'$\nu_{n}$')
    ax[0].set_xlim(0,20)
    ax[0].set_ylim(-0.5, 0.); 
    ax[0].legend(frameon=True, framealpha=0.5, facecolor='white', edgecolor='none', loc='lower right')
    ax[1].set_xlim(60,120)
    ax[1].set_ylim(-0.1, 0)
    ax[1].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
    ax[1].set_xlabel(r'$\nu_{n}$')
    
    #ax[1].plot(iw.imag, Siw_raw['up'].data[:,orb,orb].imag, 'o',ms=3, label='Dyson')
    #ax[1].plot(iw.imag, Siw_tailsm['up'].data[:,orb,orb].imag, '-',label='window1')
    #ax[1].plot(iw.imag, Siw_tailmid['up'].data[:,orb,orb].imag, '-',label='window2')
    #ax[1].plot(iw.imag, Siw_taillg['up'].data[:,orb,orb].imag, '-',label='window3')
    #ax[1].plot(iw.imag, Siw_res['up'].data[:,orb,orb].imag, '-', label='res. min.')
    #ax[1].plot(iw.imag[higher], Sigma_high00[higher].imag, '--', alpha=0.75)
    #ax[1].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
    #ax[1].set_xlabel(r'$\nu_{n}$')
    ax[0].text(3, -0.1, 'fit windows', fontsize=12, color='white')

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    #for a, let in zip([ax[0,0], ax[0,1], ax[1,0], ax[1,1]], 
    #                  ['(a)', '(b)', '(c)', '(d)']): 
    #for a, let in zip(ax, 
    #                  ['(a)', '(b)']): 
    #    t = a.text(0.03, 0.85, let, transform = a.transAxes, size=14) 
    #                              #backgroundcolor='white', alpha=0.75)
    #    #t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))
    #plt.savefig(f'sro-{name}.pdf')
    plt.savefig('sro_tailfit_windows.pdf')


