import sys

from triqs.gf import *
from triqs.operators import *
from triqs_ctseg import *
from h5 import HDFArchive
from triqs.utility import mpi 
import time
import numpy as np
np.random.seed(1234)


# single orbital hubbarD model
def _gf_fit_tail_fraction(Gf, fraction=0.4, replace=None, known_moments=None):
    """
    fits the tail of Gf object by making a polynomial
    fit of the Gf on the given fraction of the Gf mesh
    and replacing that part of the Gf by the fit

    0.4 fits the last 40% of the Gf and replaces the
    part with the tail

    Parameters
    ----------
    Gf : BlockGf (Green's function) object
    fraction: float, optional default 0.4
        fraction of the Gf to fit
    replace: float, optional default fraction
        fraction of the Gf to replace
    known_moments: np.array
        known moments as numpy array
    Returns
    -------
    Gf_fit : BlockGf (Green's function) object
            fitted Gf
    """

    Gf_fit = Gf.copy()
    # if no replace factor is given use the same fraction
    if not replace:
        replace = fraction

    for i, bl in enumerate(Gf_fit.indices):
        Gf_fit[bl].mesh.set_tail_fit_parameters(tail_fraction=fraction)
        if known_moments:
            tail = Gf_fit[bl].fit_hermitian_tail(known_moments[i])
        else:
            tail = Gf_fit[bl].fit_hermitian_tail()
        nmax_frac = int(len(Gf_fit[bl].mesh)/2 * (1-replace))
        Gf_fit[bl].replace_by_tail(tail[0],n_min=nmax_frac)

    return Gf_fit


t = 1.0; beta = 5;
mc_cycles = 1e7
n_tau = 10001
U = 4.0
S = Solver(beta=beta, gf_struct=[('up', 1), ('down', 1)], n_tau=n_tau)

S.G_iw << SemiCircular(2*t)


g = 0.5*(S.G_iw['up'] + S.G_iw['down'])
S.G_iw['up'] << g
S.G_iw['down'] << S.G_iw['up']
S.G_iw << make_hermitian(S.G_iw)
for name, g0 in S.G0_iw: g0 << inverse(iOmega_n + U/2.0 - t**2 * g)

startTime = time.time()
S.solve(h_int = U*n('up', 0)*n('down',0), 
        length_cycle=100,
        n_cycles = int(mc_cycles/mpi.size),
        n_warmup_cycles = 100000,
        measure_ft=True,
        measure_fw=False,
        measure_gt= True,
        measure_gw= False
        )
executionTime = (time.time() - startTime)
mpi.report('Execution time of solver in seconds: ' + str(executionTime))

Gf_known_moments = make_zero_tail(S.G_iw,n_moments=2)
for i, bl in enumerate(S.G_iw.indices):
    Gf_known_moments[i][1] = np.eye(S.G_iw[bl].target_shape[0])
    S.G_iw[bl] << Fourier(S.G_tau[bl], Gf_known_moments[i])

glist = [GfImTime(indices=g.indices, beta=beta, n_points=n_tau) for _, g in S.G0_iw]
S.G0_tau = BlockGf(name_list = ['up', 'down'], block_list=glist, make_copies=True)
for block, g in S.G0_iw: S.G0_tau[block] << Fourier(S.G0_iw[block])

Sigma_iw_raw = S.Sigma_iw.copy()
S.Sigma_iw << inverse(S.G0_iw) - inverse(S.G_iw)

g = 0.5*(S.Sigma_iw['up'] + S.Sigma_iw['down'])
S.Sigma_iw['up'] << g
S.Sigma_iw['down'] << S.Sigma_iw['up']

F_iw = S.Sigma_iw.copy()
Sigma_iw_IE = S.Sigma_iw.copy()
Sigma_iw_IE_direct = S.Sigma_iw.copy()
Sigma_iw_IE_fit = S.Sigma_iw.copy()
F_iw << 0.0+0.0j
Sigma_iw_IE << 0.0+0.0j
Sigma_iw_IE_direct << 0.0+0.0j
F_known_moments = make_zero_tail(F_iw, n_moments=1)


for i, bl in enumerate(F_iw.indices):
    F_iw[bl] << Fourier(S.F_tau[bl], F_known_moments[i])
# fit tail of improved estimator and G_freq

for block, fw in F_iw:
    for iw in fw.mesh:
        Sigma_iw_IE_fit[block][iw] = F_iw[block][iw] / S.G_iw[block][iw]

G_iw_fit = S.G_iw.copy()
F_iw << _gf_fit_tail_fraction(F_iw, fraction=0.9, replace=0.7, known_moments=F_known_moments)
G_iw_fit << _gf_fit_tail_fraction(G_iw_fit ,fraction=0.9, replace=0.7, known_moments=Gf_known_moments)

for block, fw in F_iw:
    for iw in fw.mesh:
        Sigma_iw_IE[block][iw] = F_iw[block][iw] / G_iw_fit[block][iw]
        # Sigma_iw_IE_direct[block][iw] = S.F_iw[block][iw] / S.G_iw[block][iw]

# if mpi.is_master_node():
#     out = 'bethe_{}_{:1.0e}_fit.h5'.format(n_tau, mc_cycles)
#     ar = HDFArchive(out, 'a')
#     ar['Sigma_iw_raw'] = Sigma_iw_raw
#     ar['Sigma_iw_fit'] = S.Sigma_iw
#     ar['Sigma_iw_IE'] = Sigma_iw_IE
#     ar['Sigma_iw_IE_fit'] = Sigma_iw_IE_fit

#     ar['G_iw']         = S.G_iw
#     ar['G_tau']        = S.G_tau

#     ar['G0_tau']       = S.G0_tau
#     ar['G0_iw']        = S.G0_iw


#     del ar