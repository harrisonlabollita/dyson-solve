import sys, glob, os
from h5 import HDFArchive
from triqs.gf import *
import scipy.integrate

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('publish')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def _Linfty_norm_G(Gloc_iw, G_tau):

    glist = [GfImTime(indices=g.indices, beta=G_tau.mesh.beta,
                      n_points=len(G_tau.mesh)) for _, g in Gloc_iw]

    Gloc_tau = BlockGf(name_list=[name for name, g in Gloc_iw], block_list=glist, make_copies=True)

    for block, gf in Gloc_iw: Gloc_tau[block] << Fourier(gf)
    norm_grid = abs(np.linalg.norm(Gloc_tau['up'].data - G_tau['up'].data, axis=tuple(range(1, G_tau['up'].data.ndim))))
    norm = np.linalg.norm(norm_grid, axis=0) * np.sqrt(1/len(G_tau.mesh))
    return norm #max([np.max(np.abs(Gloc_tau[b].data-G_tau[b].data)) for b, _ in Gloc_tau])

def Linfty_norm_G(G1, G2):
    norm_grid = abs(np.linalg.norm(G1['up'].data - G2['up'].data, axis=tuple(range(1, G1['up'].data.ndim))))
    #norm = np.linalg.norm(norm_grid, axis=0) * np.sqrt(1/len(G1.mesh))
    return np.max(norm_grid)

def add_convergence_to_plot(ax, data, **kwargs):
    mu, nlatt, nimp = [], [], []
    maxiter = min(data['dmft_results']['iterations'], 30)
    for it in range(maxiter):
        mu.append(data[f'dmft_results/it_{it}']['mu'])
        nlatt.append(data[f'dmft_results/it_{it}']['n_latt'])
        nimp.append(data[f'dmft_results/it_{it}']['n_imp'])
    mu, nimp, nlatt = np.array(mu), np.array(nimp), np.array(nlatt)
    ax[0].semilogy(list(range(1, maxiter)), np.abs(mu[:-1]-mu[-1]), 'o-', **kwargs); 
    #ax[0].semilogy(list(range(1, maxiter)), np.abs(np.diff(mu)), 'o-', **kwargs); 
    ax[0].set_ylabel(r'$\delta\mu$')
    #ax[1].semilogy(range(maxiter), np.abs(nimp-nlatt), 'o-', **kwargs); 
    #ax[1].semilogy(list(range(1,maxiter)), np.abs(np.diff(nimp)), 'o-', **kwargs); 
    ax[1].semilogy(list(range(1,maxiter)), np.abs(nimp[:-1]-nlatt[-1]), 'o-', **kwargs); 
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

    scale=1.2
    fig, ax = plt.subplots(2,1,sharex=True, figsize=(3*scale, 4*scale))
    add_convergence_to_plot(ax, HDFArchive('sro_dmft_tail_sm_wind.h5'), label='window1', color='cyan', mec='blue')
    add_convergence_to_plot(ax, HDFArchive('sro_tailfit_mc_1e8.h5'), label='window2', color='mediumslateblue', mec='purple')
    add_convergence_to_plot(ax, HDFArchive('sro_dmft_tail_lg_wind.h5'), label='window3', color='lightcoral', mec='red')
    add_convergence_to_plot(ax, res, label='CRM', color='limegreen', mec='green')
    ax[0].legend(loc='upper right', fontsize=8)
    for a, let in zip([ax[0], ax[1]], 
                      ['(a)', '(b)']): 
        t = a.text(0.03, 0.05, let, transform = a.transAxes, size=14) 
        t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))
    #plt.show()
    plt.savefig('sro-tailfit-convergence.pdf')
