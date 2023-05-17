from triqs.gf import *
from math import copysign
import numpy as np
from scipy import integrate
import sys, os, time
from h5 import HDFArchive
from triqs.utility import mpi 
from triqs_cthyb import Solver, version
from triqs.operators import c, c_dag, n, dagger
from triqs.atom_diag import trace_rho_op
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from triqs.gf import Gf, MeshImFreq, iOmega_n, inverse
from triqs.operators.util import h_int_kanamori, U_matrix_kanamori
from itertools import product
from numpy import matrix, array, block, diag, eye 
from numpy.linalg import inv 

# Get a list of all annihilation operators from a many-body operators
def get_fundamental_operators(op):
    idx_lst = []
    for term, val in op: 
        for has_dagger, (bl, orb) in term:
            if not idx_lst.count([bl, orb]):
                idx_lst.append([bl,orb])
    return [c(bl, orb) for bl, orb in idx_lst]



# ==== System Parameters ====
beta = 5.                       # Inverse temperature
mu = 0.0                        # Chemical potential
eps = array([0.0, 0.1])         # Impurity site energies
t = 0.2                         # Hopping between impurity sites

eps_bath = array([0.27, -0.4])  # Bath site energies
t_bath = 0.0                    # Hopping between bath sites

U = 1.                          # Density-density interaction
J = 0.2                         # Hunds coupling

block_names = ['up', 'dn']
n_orb = len(eps)
n_orb_bath = len(eps_bath)

# Non-interacting impurity Hamiltonian in matrix representation
h_0_mat = diag(eps - mu) - matrix([[0, t], 
                                   [t, 0]])

# Bath Hamiltonian in matrix representation
h_bath_mat = diag(eps_bath) - matrix([[0, t_bath],
                                      [t_bath, 0]])

# Coupling matrix
V_mat = matrix([[1., 1.],
                [1., 1.]])

# ==== Local Hamiltonian ====
c_dag_vec = { s: matrix([[c_dag(s,o) for o in range(n_orb)]]) for s in block_names }
c_vec =     { s: matrix([[c(s,o)] for o in range(n_orb)]) for s in block_names }

h_0 = sum(c_dag_vec[s] * h_0_mat * c_vec[s] for s in block_names)[0,0]

Umat, Upmat = U_matrix_kanamori(n_orb, U_int=U, J_hund=J)
h_int = h_int_kanamori(block_names, range(n_orb), Umat, Upmat, J, off_diag=True)

h_imp = h_0 + h_int

# ==== Bath & Coupling Hamiltonian ====
c_dag_bath_vec = { s: matrix([[c_dag(s, o) for o in range(n_orb, n_orb + n_orb_bath)]]) for s in block_names }
c_bath_vec =     { s: matrix([[c(s, o)] for o in range(n_orb, n_orb + n_orb_bath)]) for s in block_names }

h_bath = sum(c_dag_bath_vec[s] * h_bath_mat * c_bath_vec[s] for s in block_names)[0,0]
h_coup = sum(c_dag_vec[s] * V_mat * c_bath_vec[s] + c_dag_bath_vec[s] * V_mat.H * c_vec[s] for s in block_names)[0,0]

# ==== Total impurity Hamiltonian ====
h_tot = h_imp + h_coup + h_bath

# ==== Green function structure ====
gf_struct = [ (s, n_orb) for s in block_names ]

# ==== Non-Interacting Impurity Green function  ====
n_iw = int(100 * beta)
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
G0_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
h_tot_mat = block([[h_0_mat, V_mat     ],  
                   [V_mat.H, h_bath_mat]])
for bl, iw in product(block_names, iw_mesh):
    G0_iw[bl][iw] = inv(iw.value * eye(2*n_orb) - h_tot_mat)[:n_orb, :n_orb]

ar = HDFArchive('pyed.h5')
G_iw_ref = ar['G']
del ar
G0_tau_ref = make_gf_from_fourier(G0_iw)
G_tau_ref = make_gf_from_fourier(G_iw_ref)
Sigma_iw_ref = inverse(G0_iw) - inverse(G_iw_ref)

ar = HDFArchive('cthyb-pm.h5')
dm = ar['dm']
hdiag = ar['h_loc_diag']
G_iw_data = ar['G']
G_tau_data = ar['Gtau']
del ar

def anticomm(A,B): return A*B + B*A
def comm(A,B): return A*B - B*A
def sigma_high_frequency_moments(density_matrix,
                           ad_imp, 
                           gf_struct, 
                           h_int):
    """
    Calculate the first and second high frequency moment of Sigma_iw
    following Rev. Mod. Phys. 83, 349 (2011). They read
    (0) Sigma_0       = -<{[Hint,c],c+}> (Hartree shift)
    (1) Sigma_1       =  <{[Hint,[Hint,c]],c+}> - Sigma_0^2,
    where Hint is the interaction Hamiltonian
    Parameters
    ----------
    density_matrix : list, np.ndarray
                     measured density matrix from TRIQS/CTHYB.
    ad_imp         : AtomDiag
                     h_loc_diagonalization from TRIQS/CTHYB.
    gf_struct      : List of pairs (str,int)
                     Block structure of Green's function.
    h_int          : triqs.operators.Operator
                     interaction Hamiltonian
    Returns
    -------
    sigma_moments  : dict, np.ndarray
                     first and second moments in a dict with the
                     same block strucutre of the TRIQS Gf object.
    """


    sigma_moments = {bl : np.zeros((2, bl_size, bl_size),dtype=complex) for bl, bl_size in gf_struct}
    for bl, bl_size in gf_struct:
        for orb1 in range(bl_size):
            for orb2 in range(bl_size):

                # Sigma_HF term
                op_HF = -anticomm(comm(h_int, c(bl,orb1)), c_dag(bl,orb2))
                sigma_moments[bl][0,orb1,orb2] = trace_rho_op(density_matrix, op_HF, ad_imp)

                # Sigma_1/iwn term
                op_iw = anticomm(comm(h_int, comm(h_int, c(bl,orb1))), c_dag(bl,orb2))
                sigma_moments[bl][1,orb1,orb2] = trace_rho_op(density_matrix, op_iw, ad_imp) - sigma_moments[bl][0,orb1,orb2]**2

    return sigma_moments
