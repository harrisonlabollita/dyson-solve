import sys
from triqs.gf import *
from h5 import HDFArchive
import numpy as np
np.random.seed(1)

from dimer import *

import matplotlib.pyplot as plt
try:
    plt.style.use('publish')
except:
    pass

sys.path.append('../../')
from dyson_solve import Dyson

fig, ax = plt.subplots(2,3, figsize=(12,5))
tolerances = [1e-4, 1e-8, 1e-12] 
string_tol = ['1e-4', '1e-8', '1e-12'] 

for itol, tol in enumerate(tolerances):

    G_tau_qmc= G_tau_ref.copy()
    beta = G_tau_qmc.mesh.beta
    for block, _ in G_tau_qmc: G_tau_qmc[block].data[:] += tol*(2*np.random.rand(*G_tau_qmc[block].data.shape)-1)
    G_iw_qmc = make_gf_from_fourier(G_tau_qmc)

    dys = Dyson(lamb=30, eps=tol, options=dict(maxiter=5000, disp=True) )
    sigma_moments = sigma_high_frequency_moments(dm, hdiag, gf_struct, h_int)

    tau_i = np.array([float(x) for x in G_tau_qmc.mesh])
    iw_i = np.array([complex(x) for x in G_iw_qmc.mesh])

    g_xaa = dys.d.lstsq_dlr_from_tau(tau_i, G_tau_qmc['up'].data, beta)
    g_tau=dys.d.eval_dlr_tau(g_xaa, tau_i, beta)
    g_iw=dys.d.eval_dlr_freq(g_xaa, iw_i, beta)

    Sigma_iw_raw = Sigma_iw_ref.copy()
    Sigma_iw_raw.zero();

    result = dys.solve(Sigma_iw = Sigma_iw_raw,
                       G0_tau = G0_tau_ref,
                       G_tau=G_tau_qmc,
                       Sigma_moments = sigma_moments,       
                       beta = beta,
                       )
    Sigma_iw_fit = result.Sigma_iw
    history = result.dlr_optim['up'].callback


    if tol == 1e-4:
        ax[0,0].plot(tau_i/beta, g_tau[:,0,0].real, label='DLR')
        ax[0,0].plot(tau_i/beta, G_tau_qmc['up'].data[:,0,0].real, ls='--', label='QMC')
        ax[0,0].legend()
        ax[0,0].set_ylabel(r'$G(\tau)$')
        ax[0,0].set_xlabel(r'$\tau/\beta$')


        ax[0,1].plot(iw_i.imag, g_iw[:,0,0].imag, label='DLR')
        ax[0,1].plot(iw_i.imag, G_iw_qmc['up'].data[:,0,0].imag, label='QMC', ls='--')
        ax[0,1].set_ylabel(r'Im$G(i\nu_{n})$')
        ax[0,1].set_xlabel(r'$\nu_{n}$')
        ax[0,1].set_xlim(-40, 40)

        freq = dys.d.get_matsubara_frequencies(beta)
        dyson   = inverse(make_gf_from_fourier(G0_tau_ref)) -  inverse(G_iw_qmc)

        convert = lambda f : list(map(int, 0.5*(f.imag*beta/np.pi - 1)))

        dyson = np.array([dyson['up'](f)[0,0] for f in convert(freq)])

        converged = history[-1][-2]

        ax[0,2].plot(freq.imag, dyson.imag, 'o-', label=r'$G_{0}^{-1}-G^{-1}$')
        ax[0,2].plot(freq.imag, converged[:,0,0].imag, 'o-', mfc='none', label=r'converged')
        lower=np.where(iw_i.imag < -5) 
        upper=np.where(iw_i.imag > 5)

        ax[0,2].plot(iw_i.imag, Sigma_iw_ref['up'].data[:,0,0].imag, label='ref')
        ax[0,2].plot(iw_i.imag[lower], (sigma_moments['up'][1][0,0]/iw_i[lower]).imag, ls='--', color='tab:red', alpha=0.75, 
                label=r'$\Sigma_{1}/i\nu_{n}$')
        ax[0,2].plot(iw_i.imag[upper], (sigma_moments['up'][1][0,0]/iw_i[upper]).imag, ls='--', color='tab:red', alpha=0.75, )
        ax[0,2].set_ylim(-0.05, 0.05)
        ax[0,2].set_ylabel(r'Im$\Sigma(i\nu_{n})$')
        ax[0,2].set_xlabel(r'$\nu_{n}$')
        ax[0,2].set_xlim(-40, 40)

    ax[1,0].semilogy(tau_i/beta, np.abs(G_tau_qmc['up'].data[:,0,0]-g_tau[:,0,0]), label=r'$\eta = $'+string_tol[itol])
    ax[1,0].set_ylabel(r'$|G(\tau)-G_{\mathrm{QMC}}(\tau)|$', fontsize=12)
    ax[1,0].set_xlabel(r'$\tau/\beta$')
    ax[1,0].set_ylim(1e-14, 1e-1)
    ax[1,1].semilogy(iw_i.imag, np.abs(G_iw_qmc['up'].data[:,0,0]-g_iw[:,0,0]))
    ax[1,1].set_ylabel(r'$|G(i\nu_{n})-G_{\mathrm{QMC}}(i\nu_{n})$|', fontsize=12,)
    ax[1,1].set_xlabel(r'$\nu_{n}$')
    ax[1,1].set_ylim(1e-14, 1e-1)
    ax[1,1].set_xlim(-40, 40)


    ax[1,2].semilogy(iw_i.imag, np.abs(Sigma_iw_ref['up'].data[:,0,0]-Sigma_iw_fit['up'].data[:,0,0]))
    ax[1,2].set_ylabel(r'$|\Sigma(i\nu_{n})-\Sigma_{\mathrm{ref}}(i\nu_{n})|$', fontsize=12)
    ax[1,2].set_xlabel(r'$\nu_{n}$')
    #ax[1,2].set_xlabel(r'$\tau$')
    ax[1,2].set_ylim(1e-14, 1e-1)
    ax[1,2].set_xlim(-40, 40)
    #ax[1,2].set_xlim(0,1)
    #ax[1,2].legend(frameon=True, framealpha=0.75, facecolor='white', edgecolor='none')

ax[0,2].legend(fontsize=9, loc='lower left')
ax[1,0].legend(frameon=True, framealpha=0.75, facecolor='white', edgecolor='none', fontsize=9, loc='upper right')
for a, let in zip([ax[0,0], ax[0,1], ax[0,2],
                   ax[1,0], ax[1,1], ax[1,2]], 
                   ['(a)', '(b)', '(c)', 
                    '(d)', '(e)', '(f)']): 
                       t = a.text(0.03, 0.85, let, transform = a.transAxes, size=14) 
                                                  #backgroundcolor='white', alpha=0.75)
                       t.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.75, lw=0))
plt.subplots_adjust(hspace=0.4, wspace=0.5)
plt.show()#plt.savefig('fig_dimer_example.pdf')#plt.show()
