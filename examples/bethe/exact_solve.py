import sys, os

from pydlr import dlr, kernel
import numpy as np

import matplotlib.pyplot as plt
try: plt.style.use('publish') 
except: pass

from triqs.gf import *
from h5 import HDFArchive

import scipy

cmd = lambda name, default : int(os.getenv(name)) if os.getenv(name) is not None else default

#USE_IW  = cmd('USE_IW', 0)
#USE_TAU = cmd('USE_TAU', 1-USE_IW)
CMP = cmd('CMP', 0)


def iw_kernel(iw, omega):
    return (iw[:,None] + omega[None, :])**(-1)

def construct_Mkl(d):
    Mkl = np.zeros((len(d), len(d)), dtype=complex)
    for iwk, wk in enumerate(d.dlrrf):
        for iwl, wl in enumerate(d.dlrrf):
            K0wk, Kbwk = kernel(np.array([0.,1.]), np.array([wk]))
            K0wl, Kbwl = kernel(np.array([0.,1.]), np.array([wl]))
            if np.fabs(wk+wl) < 1e-13: 
                Mkl[iwk,iwl] = K0wk*K0wl
            else: 
                Mkl[iwk, iwl] = (K0wk*K0wl - Kbwk*Kbwl)
                Mkl[iwk, iwl] /= (wk+wl)
    return Mkl


def fit_with_constraint(d, tau, G, beta):
    K = kernel(tau/beta, d.dlrrf)         # (m, n)
    one = np.ones((1, len(d)))            # (p, n)
    G = G.flatten()                       # m
    constraint = np.array([-1])           # p
    results = scipy.linalg.lapack.zgglse(K, one, G, constraint)
    return results[-2]



def schur_complement_solve(d, beta, Sigma_moments, freq_mesh,
                           G_tau  = None,
                           G0_tau = None,
                           G_iw   = None,
                           G0_iw  = None,
                          ):

    Mkl = construct_Mkl(d)

    if G_tau is not None and G0_tau is not None:
    
        tau_mesh = np.array([float(x) for x in G_tau.mesh])
        #g_xaa    = d.lstsq_dlr_from_tau(tau_mesh, G_tau.data, beta)
        #g0_xaa   = d.lstsq_dlr_from_tau(tau_mesh, G0_tau.data, beta)
        g_xaa    = fit_with_constraint(d, tau_mesh, G_tau.data, beta)
        g0_xaa   = fit_with_constraint(d, tau_mesh, G0_tau.data, beta)

        # compute and obtain initial Σ
        g_iwaa  = d.matsubara_from_dlr(g_xaa, beta).flatten()
        g0_iwaa = d.matsubara_from_dlr(g0_xaa, beta).flatten()

    if G_iw is not None and G0_iw is not None:

        mesh = np.array([complex(x) for x in G_iw.mesh])
        g_xaa    = d.lstsq_dlr_from_matsubara(mesh, G_iw.data, beta)
        g0_xaa   = d.lstsq_dlr_from_matsubara(mesh, G0_iw.data, beta)

        # compute and obtain initial Σ
        g_iwaa  = d.matsubara_from_dlr(g_xaa, beta).flatten()
        g0_iwaa = d.matsubara_from_dlr(g0_xaa, beta).flatten()

    result = {}
    
    tau_k = d.get_tau(beta)/beta
    inu_k = d.get_matsubara_frequencies(beta)
    wl    = d.dlrrf
    
    Sigma_infty, Sigma_1 = Sigma_moments[0][0,0], Sigma_moments[1][0,0]
    Ktilde     = -iw_kernel(inu_k, wl/beta)
    
    one        = np.ones(len(d)).reshape(-1, 1) 
    Ktilde_mT1 = np.linalg.solve(Ktilde.T, one)
    
    Dtilde = np.diag((g0_iwaa*g_iwaa))
    
    Sigma1_tilde = one.T @ np.linalg.solve(Ktilde, (1/g0_iwaa - 1/g_iwaa))
    
    constraint = Sigma_infty*(one.T@Ktilde_mT1) - Sigma_1
    
    u = Ktilde.T@(np.linalg.solve(Dtilde, Ktilde_mT1))
    
    M1u = np.linalg.solve(Mkl, u)
    
    h = (g_iwaa - g0_iwaa)
    
    rho = h + ((constraint[0,0] - Sigma1_tilde[0])/(u.T@M1u))[0,0] * (Ktilde@M1u).flatten()
    
    sigma = np.linalg.solve(Dtilde, rho)
    
    sig_xaa = np.linalg.solve(Ktilde, sigma - Sigma_infty)
    sigma_iw = d.eval_dlr_freq(sig_xaa, freq_mesh, beta) + Sigma_infty

    return sigma_iw

lambs     =  [1, 2, 4, 6, 8, 10, 12, 15, 20, 22, 25, 27, 30, 40, 50]
lamb_keys = { lambs[i] : i for i in range(len(lambs)) }

mc_cycles =  [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9]
mc_cycles_keys = { mc_cycles[i] : i for i in range(len(mc_cycles)) }

ref_file = 'data_beta_5/bethe_0_10001_1e+11.h5'
ar = HDFArchive(ref_file)
Sigma_ref = ar['Sigma_iw']['up']
freq_mesh = np.array([complex(x) for x in Sigma_ref.mesh], dtype=complex)
Sigma_ref = ar['Sigma_iw']['up'].data.flatten()
sig_max = max(np.abs(Sigma_ref))
del ar


def exp(USE_TAU=1, USE_IW=0):

    diff_optim = np.zeros((len(lambs), len(mc_cycles), len(Sigma_ref)))
    optim = np.zeros((len(lambs), len(mc_cycles), len(Sigma_ref)), dtype=complex)

    diff_solve = np.zeros((len(lambs), len(mc_cycles), len(Sigma_ref)))
    solve = np.zeros((len(lambs), len(mc_cycles), len(Sigma_ref)), dtype=complex)

    #diff_dyson = np.zeros((len(lambs), len(mc_cycles), len(Sigma_ref)))
    #dyson = np.zeros((len(lambs), len(mc_cycles), len(Sigma_ref)), dtype=complex)

    for il, lamb in enumerate(lambs):
        for im, mc in enumerate(mc_cycles):

            file = 'data_beta_5/bethe_{}_10001_{:1.0e}.h5'.format(lamb, mc)
            ar = HDFArchive(file)

            Sigma_optim = ar['Sigma_iw_fit']['up']
            freq_mesh = np.array([complex(x) for x in Sigma_optim.mesh], dtype=complex)
            Sigma_optim = ar['Sigma_iw_fit']['up'].data.flatten()

            Sigma_moments = ar['Sigma_moments']['up']

            if USE_TAU:

                G    = ar['G_tau']['up']
                beta = G.mesh.beta
                G0   = ar['G0_tau']['up']

                d = dlr(lamb=lamb, eps=1e-6)
                Sigma_solve = schur_complement_solve(d, beta, Sigma_moments, freq_mesh,
                                                    G_tau = G,
                                                    G0_tau = G0
                                                    )

            if USE_IW:
                G    = ar['G_iw']['up']
                beta = G.mesh.beta
                G0   = ar['G0_iw']['up']

                d = dlr(lamb=lamb, eps=1e-6)
                Sigma_solve = schur_complement_solve(d, beta, Sigma_moments, freq_mesh,
                                                    G_iw = G,
                                                    G0_iw = G0
                                                    )

            diff_optim[il,im, :] = np.abs(Sigma_optim-Sigma_ref)
            optim[il,im, :] = Sigma_optim
            diff_solve[il,im, :] = np.abs(Sigma_solve-Sigma_ref)
            solve[il,im, :] = Sigma_solve

    return diff_optim, optim, diff_solve, solve


if __name__ == "__main__":

    for setting in [1, 0]:
        num = f"{'iw' if 1-setting else 'tau'}"
        diff_optim, optim, diff_solve, solve = exp(USE_TAU=setting, USE_IW=1-setting)

        if CMP >= 0:

            fig,ax = plt.subplots(2,2, sharey=True, num=num)

            ax[0,0].set_title('scipy optimizer')
            ax[1,0].set_title('numpy solver')
            ax[0,1].set_title('scipy optimizer')
            ax[1,1].set_title('numpy solver')

            for key in [5e3, 5e5, 5e7, 1e8, 5e9]:
                ax[0,0].semilogy(lambs,     np.max(diff_optim[:,mc_cycles_keys[key]], axis=1)/sig_max, 'o-', label='{:1.0e}'.format(key))
                ax[1,0].semilogy(lambs,     np.max(diff_solve[:,mc_cycles_keys[key]], axis=1)/sig_max, 'o-', label='{:1.0e}'.format(key))
            ax[0,0].legend(ncols=2)

            for key in [4, 8, 12, 20, 40]:
                ax[0,1].loglog(mc_cycles, np.max(diff_optim[lamb_keys[key],:], axis=1)/sig_max, 'o-', label=str(key))
                ax[1,1].loglog(mc_cycles, np.max(diff_solve[lamb_keys[key],:], axis=1)/sig_max, 'o-', label=str(key))

            ax[0,1].legend(ncols=2)

            plt.subplots_adjust(hspace=0.3, wspace=0.3)

        if CMP >= 1:

            lamb = 20
            mc  =  5e9

            fig, ax = plt.subplots(3,1, sharex=True, num="Lamb = {}, mc = {:1.0e}, mesh = {}".format(lamb, mc, num))

            ax[0].plot(freq_mesh.imag, optim[lamb_keys[lamb], mc_cycles_keys[mc]].real, 'o', mec='k', color='tab:blue')
            ax[0].plot(freq_mesh.imag, solve[lamb_keys[lamb], mc_cycles_keys[mc]].real, 'x', color='tab:red')

            ax[1].plot(freq_mesh.imag, optim[lamb_keys[lamb], mc_cycles_keys[mc]].imag, 'o', mec='k', color='tab:blue')
            ax[1].plot(freq_mesh.imag, solve[lamb_keys[lamb], mc_cycles_keys[mc]].imag, 'x', color='tab:red')

            ax[2].semilogy(freq_mesh.imag, np.abs(optim[lamb_keys[lamb], mc_cycles_keys[mc]]-Sigma_ref), color='tab:blue', label='optim-ref')
            ax[2].semilogy(freq_mesh.imag, np.abs(solve[lamb_keys[lamb], mc_cycles_keys[mc]]-Sigma_ref), color='tab:red', label='solve-ref', ls='--')

            ax[2].legend()
            ax[-1].set_xlabel(r'$\nu_{n}$')
            ax[1].set_xlim(-100, 100)
            ax[1].set_ylim(-2, 2); ax[0].set_ylim(1.5,2.5)

        if CMP > 1:

            lamb = 20
            mc  =  1e6

            fig, ax = plt.subplots(3,1, sharex=True, num="Lamb = {}, mc = {:1.0e}, mesh = {}".format(lamb, mc, num))

            ax[0].plot(freq_mesh.imag, optim[lamb_keys[lamb], mc_cycles_keys[mc]].real, 'o', mec='k', color='tab:blue')
            ax[0].plot(freq_mesh.imag, solve[lamb_keys[lamb], mc_cycles_keys[mc]].real, 'x', color='tab:red')

            ax[1].plot(freq_mesh.imag, optim[lamb_keys[lamb], mc_cycles_keys[mc]].imag, 'o', mec='k', color='tab:blue')
            ax[1].plot(freq_mesh.imag, solve[lamb_keys[lamb], mc_cycles_keys[mc]].imag, 'x', color='tab:red')

            ax[2].semilogy(freq_mesh.imag, np.abs(optim[lamb_keys[lamb], mc_cycles_keys[mc]]-Sigma_ref), color='tab:blue', label='optim-ref')
            ax[2].semilogy(freq_mesh.imag, np.abs(solve[lamb_keys[lamb], mc_cycles_keys[mc]]-Sigma_ref), color='tab:red', label='solve-ref', ls='--')

            ax[2].legend()
            ax[-1].set_xlabel(r'$\nu_{n}$')
            ax[1].set_xlim(-100, 100)
            ax[1].set_ylim(-2, 2); ax[0].set_ylim(1.5,2.5)

    plt.show()
