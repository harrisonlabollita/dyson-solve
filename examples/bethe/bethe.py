import sys

from triqs.gf import *
from triqs.operators import *
from triqs_cthyb import *
from h5 import HDFArchive
from triqs.utility import mpi 

import numpy as np
np.random.seed(1234)

sys.path.append('../../')
from dyson_solve import Dyson


# single orbital hubbar model
def run_bethe(lamb, n_tau, mc_cycles, eps=1e-6):

    t = 1.0; beta = 10;

    U = 4.0
    S = Solver(beta=beta, gf_struct=[('up', 1), ('down', 1)], n_tau=n_tau)

    S.G_iw << SemiCircular(2*t)
    S.Sigma_iw = S.G_iw.copy()
    S.Sigma_iw.zero();
    S.Sigma_iw << 0.5*U
    
    # Dyson solver object
    dys = Dyson(lamb=lamb, eps=eps)

    g = 0.5*(S.G_iw['up'] + S.G_iw['down'])
    S.G_iw['up'] << g
    S.G_iw['down'] << S.G_iw['up']
    S.G_iw << make_hermitian(S.G_iw)
    for name, g0 in S.G0_iw: g0 << inverse(iOmega_n + U/2.0 - t**2 * g)
    
    S.solve(h_int = U*n('up', 0)*n('down',0), 
            length_cycle=100,
            n_cycles = int(mc_cycles/mpi.size),
            n_warmup_cycles = 10000,
            measure_density_matrix = True,
            use_norm_as_weight = True,
            off_diag_threshold = 1e-5,
            move_double=True,
            move_shift=False,
           )

    
    glist = [GfImTime(indices=g.indices, beta=beta, n_points=S.n_tau) for _, g in S.G0_iw]
    S.G0_tau = BlockGf(name_list = ['up', 'down'], block_list=glist, make_copies=True)
    for block, g in S.G0_iw: S.G0_tau[block] << Fourier(S.G0_iw[block])

    if mpi.is_master_node():
        Sigma_iw_raw = S.Sigma_iw.copy()
        Sigma_iw_fit = dys.solve(Sigma_iw_raw,
                                 S.G_tau, 
                                 S.G0_tau,
                                 S.Sigma_moments
                                 )

        S.Sigma_iw << Sigma_iw_fit

    S.Sigma_iw = mpi.bcast(S.Sigma_iw)
    g = 0.5*(S.Sigma_iw['up'] + S.Sigma_iw['down'])
    S.Sigma_iw['up'] << g
    S.Sigma_iw['down'] << S.Sigma_iw['up']

    for name, sig in S.Sigma_iw: S.Sigma_iw[name] << make_hermitian(sig)
    S.G_iw << inverse(inverse(S.G0_iw) - S.Sigma_iw)

    if mpi.is_master_node():
        out = 'bethe_{}_{}_{:1.0e}.h5'.format(lamb, n_tau, mc_cycles)
        ar = HDFArchive(out, 'a')
        ar['Sigma_iw_raw'] = Sigma_iw_raw
        ar['Sigma_iw_fit'] = S.Sigma_iw
        ar['Sigma_moments'] = S.Sigma_moments

        ar['G_iw']         = S.G_iw
        ar['G_tau']        = S.G_tau

        ar['G0_tau']       = S.G0_tau
        ar['G0_iw']        = S.G0_iw

        ar['Solver']       = S

        del ar
