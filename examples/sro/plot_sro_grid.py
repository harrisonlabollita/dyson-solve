import numpy as np, matplotlib.pyplot as plt
from h5 import HDFArchive

from triqs.gf import BlockGf, Gf

try: plt.style.use('publish')
except: pass

if __name__ == "__main__":
    
    data = HDFArchive('sro_200_1e-06_tau_10001_mc_1e8.h5')
    Sigma = data['dmft_results']['last_iter']['Sigma_iw']
    const = Sigma['up'](0).real
    iw, sig = Sigma['up'].x_data_view();


    Sigma_moments = data['dmft_results']['last_iter']['Sigma_moments']
    sig_infty, sig_1  = Sigma_moments['up'][0], Sigma_moments['up'][1]
    hf_range = iw[np.where(iw.imag > 6)]
    sig_hf = sig_infty + sig_1/hf_range[:,None, None]

    m, n = sig.shape[1:]
    fig,ax = plt.subplots(3,3, sharex=True, sharey=True)

    for i in range(m):
        for j in range(n):

            txt = r'$\Sigma_{\infty}$ = ' +  '{:.3f}'.format(sig_infty[i,j].real) + ', ' + r'$\Sigma_{1}$ = ' + '{:.3f}'.format(sig_1[i,j].real)
            ax[i,j].text(5, 1/32, txt, fontsize=8)
            ax[i,j].plot(iw.imag, sig[:,i,j].real-const[i,j], lw=2, color='tab:blue', label='Re')
            ax[i,j].plot(iw.imag, sig[:,i,j].imag, lw=2, color='tab:red', label='Im')
            ax[i,j].plot(hf_range.imag, sig_hf[:,i,j].real-const[i,j], '--', lw=1, color='k', label=r'$\Sigma_{\infty}$')
            ax[i,j].plot(hf_range.imag, sig_hf[:,i,j].imag,            '-.', lw=1, color='k', label=r'$\Sigma_{1}/i\nu_{n}$')
            ax[i,j].set_xlim(0, 30)
            ax[i,j].set_ylim(-1/2, 1/8)
    h, l = ax[0,0].get_legend_handles_labels()
    fig.legend(h, l, ncols=len(h), loc='upper center')
    for i in range(3): ax[-1, i].set_xlabel(r'$\nu_{n}$')
    ax[1,0].set_ylabel(r'$\Sigma^{ab}(i\nu_{n})$')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()
