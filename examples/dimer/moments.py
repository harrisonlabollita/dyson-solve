from triqs.gf import *
from h5 import HDFArchive
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dimer import *

import matplotlib.pyplot as plt
try: plt.style.use('publish')
except: pass

sigma_moments = sigma_high_frequency_moments(dm, hdiag, gf_struct, h_int)

iwn = np.array([complex(x) for x in Sigma_iw_ref.mesh])

compute_tail = lambda mom :  mom[0] + mom[1]/iwn

fig, ax = plt.subplots(1,2, figsize=(6, 3))

tail = compute_tail(sigma_moments['up'][:,0,0])

axins = inset_axes(ax[0], width="55%", height="35%", borderpad=1.5, loc=4)
axins.tick_params(labelsize=8)
for a in (ax[0], axins):
    a.plot(iwn.imag, Sigma_iw_ref['up'][0,0].data.real*1000, 'o', ms=4, mfc='none', label=r'$\Sigma$')
    a.plot(iwn.imag, tail.real*1000, lw=1, label=r'$\Sigma_{\infty} +  \frac{1}{i\nu_{n}}\Sigma_{1}$')
ax[0].legend(loc='upper left', ncols=2)
ax[0].set_xlabel(r'$\nu_{n}$'); ax[0].set_ylabel(r'Re$\Sigma^{0,0}$ (meV)')
ax[0].set_xlim(0, 100); ax[0].set_ylim(460, 490)
axins.set_xlim(200,400); axins.set_ylim(483.75, 484.1)

axins = inset_axes(ax[1], width="55%", height="35%", borderpad=1.5, loc=4)
axins.tick_params(labelsize=8)
for a in (ax[1], axins):
    a.plot(iwn.imag, Sigma_iw_ref['up'][1,1].data.imag*1000, 'o', ms=4, mfc='none')
    a.plot(iwn.imag, tail.imag*1000, lw=1)
ax[1].set_xlabel(r'$\nu_{n}$'); ax[1].set_ylabel(r'Im$\Sigma^{0,0}$ (meV)')
ax[1].tick_params(labelright=True, labelleft=False)
ax[1].yaxis.set_label_position('right')
ax[1].set_xlim(0, 100); ax[1].set_ylim(-0.03*1000, 0)
axins.set_xlim(200,400); axins.set_ylim(-0.0015*1000, -0.0005*1000)
plt.subplots_adjust(wspace=0.1)
plt.savefig('fig_dimer_moments.pdf')
