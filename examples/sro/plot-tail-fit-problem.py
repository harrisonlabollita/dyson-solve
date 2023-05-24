from triqs.gf import *
from h5 import HDFArchive
import numpy as np
import matplotlib.pyplot as plt
try: plt.style.use('publish')
except: pass

filename = 'sro_200_1e-06_tau_10001_mc_1e8.h5'
ar = HDFArchive(filename)
Sigma_raw = ar['dmft_results/it_0']['Sigma_iw_raw']
del ar


iw = np.array([complex(x) for x in Sigma_raw.mesh])
plt.figure(figsize=(5,3))
plt.plot(iw.imag, Sigma_raw['up'].data[:,0,0].imag, 'o', ms=4, label=r'$G_{0}^{-1}-G^{-1}$')

shape=[0] + list(Sigma_raw['up'].target_shape)
min_w, max_w = 4, 7
tail, err = Sigma_raw['up'].fit_hermitian_tail_on_window(n_min=int(((40/np.pi)*min_w -1)/2),
                                                         n_max=int(((40/np.pi)*max_w -1)/2),
                                                         known_moments=np.zeros(shape,dtype=complex),
                                                         n_tail_max=2*len(Sigma_raw.mesh),
                                                         expansion_order=4)
#for m in tail : print(m[0,0])

Sigma_tail = Sigma_raw.copy()
Sigma_tail['up'].replace_by_tail(tail, int(((40/np.pi)*max_w -1)/2))

high_tail = lambda coef :  coef[0][0,0] + coef[1][0,0]/iw + coef[2][0,0]/iw/iw + coef[3][0,0]/iw/iw/iw+coef[4][0,0]/iw/iw/iw/iw

plt.plot(iw.imag, Sigma_tail['up'].data[:,0,0].imag, label=r'tail fitted')
plt.plot(iw.imag, high_tail(tail).imag, ls='--', alpha=0.75, label=r'$\sum_{j}\frac{c_{j}}{(i\nu_{n})^{j}}$ (tail)')
plt.text(max_w-1, -0.375, 'fitting window', color='grey', rotation=90)
plt.axvspan(min_w,max_w, color='grey', alpha=0.25, lw=0)
plt.xlim(0,30); plt.ylim(-0.4, 0.1)
plt.legend(); plt.xlabel(r'$\nu_{n}$')
plt.ylabel(r'Im$\Sigma(i\nu_{n})$')
plt.show() #plt.savefig('intro.pdf')
