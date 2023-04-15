import sys, os

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('publish')

from triqs.gf import *
from h5 import HDFArchive


parent_dir = 'data'

file_base = 'bethe_{}_{}_{:1.0e}.h5'

lambdas   =  [20, 40, 60, 100, 200, 400]
mc_cycles =  [1e5, 1e6, 1e7, 1e8, 1e9]
n_taus    =  [5001, 10001, 20001, 50001] 



def plot_mc_vs_tau_grid():

    for lamb in lambdas:
        fig, ax = plt.subplots(len(mc_cycles), len(n_taus), sharex=True, sharey=True)
        fig.suptitle(r'$\Lambda$ = {}'.format(lamb))

        for i,mc in enumerate(mc_cycles):
            for j, nt in enumerate(n_taus):
                ar = HDFArchive(parent_dir+'/'+file_base.format(lamb, nt, mc))
                S_fit =  ar['Sigma_iw_fit']
                S_raw =  ar['Sigma_iw_raw']
                om_mesh = np.array([complex(x) for x in S_fit.mesh])

                ax[i,j].plot(om_mesh.imag, S_raw['up'].data.flatten().imag)
                ax[i,j].plot(om_mesh.imag, S_fit['up'].data.flatten().imag)
                ax[i,j].set_title('{:1.1e}, {}'.format(mc, nt))
                ax[i,j].set_xlim(0, 30)
                ax[i,j].set_ylim(-1, 0)
    plt.show()

if __name__ == "__main__":
    plot_mc_vs_tau_grid()
