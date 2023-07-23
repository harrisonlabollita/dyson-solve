from typing import NamedTuple, Any

import numpy as np
from pydlr import kernel, dlr
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

from triqs.gf import *

is_block_gf = lambda x : isinstance(x, BlockGf)
is_array = lambda x : isinstance(x, np.ndarray)

class CallbackResult(NamedTuple):
    x  : Any      = None
    sigma : Any   = None
    residual : Any = None

class MinimizerResult(NamedTuple):
    scipy_sol : Any = None
    sig_xaa : Any  = None
    g_xaa   : Any  = None
    g0_xaa  : Any  = None
    callback : CallbackResult = None

class SolverResult(NamedTuple):
    Sigma_iw  : Any    = None
    G     : Any    = None
    G0    : Any    = None
    Sigma_moments : Any = None
    minimizer : MinimizerResult     = None

class Callback:

    def __init__(self, eval_func, diff_func, method):
        self._history   = [] # stores (x⃗, Σiνₖ, G-G₀-G₀ΣG) in a CallbackResult
        self._eval_func = eval_func
        self._diff_func = diff_func
        self._method     = method

    def __call__(self, *args):
        x = args[0]
        self._history.append(CallbackResult(x=x, sigma=self._eval_func(x),
                                             residual=self._diff_func(x)))
    @property
    def history(self): return self._history


class Symmetrizer:

    def __init__(self, nx, no):
        self.N = (no*(no-1))//2
        self.nx, self.no = nx, no
        self.diag_idxs = np.arange(self.no)
        self.triu_idxs = np.triu_indices(no, k=1)
        self.tril_idxs = np.tril_indices(no, k=-1)
    
    def get_x_d(self, g_xaa):
        x_d = g_xaa[:, self.diag_idxs, self.diag_idxs].flatten()
        return x_d

    def set_x_d(self, g_xaa, x_d):
        g_xaa[:, self.diag_idxs, self.diag_idxs] = x_d.reshape((self.nx, self.no))
        return g_xaa

    def get_x_u(self, g_xaa):
        x_u = g_xaa[:, self.triu_idxs[0], self.triu_idxs[1]].flatten()
        return x_u

    def set_x_u(self, g_xaa, x_u):
        g_xaa[:, self.triu_idxs[0], self.triu_idxs[1]] = x_u.reshape((self.nx, self.N))
        g_xaa[:, self.tril_idxs[0], self.tril_idxs[1]] = g_xaa[:, self.triu_idxs[0], self.triu_idxs[1]].conj()
        return g_xaa
    
    def get_x_l(self, g_xaa):
        x_l = g_xaa[:, self.tril_idxs[0], self.tril_idxs[1]].flatten()
        return x_l
    
    def set_x_l(self, g_xaa, x_l):
        g_xaa[:, self.tril_idxs[0], self.tril_idxs[1]] = x_l.reshape((self.nx, self.N))
        return g_xaa
        
    def get_diag_indices(self): return self.diag_idxs
    def get_triu_indices(self): return self.triu_idxs



class Dyson(object):

    def __init__(self, lamb, 
                       eps=1e-9,                
                       method='trust-constr', 
                       options=dict(maxiter=10000),
                       verbose=True,
                       **kwargs
                       ):

        self.lamb    = lamb                              # dlr lambda
        self.eps     = eps                               # dlr epsilon
        self._dlr    = dlr(lamb=self.lamb, eps=self.eps, **kwargs) # dlr class instance
        self.Mkl     = self._compute_mkl()               # compute residual matrix
        self.method  = method                          # scipy minimize method
        self.options = options                        # minimize options
        self.verbose = verbose

        # if verbose add display to minimize options
        if self.verbose: self.options['disp'] = True
        else: self.options['disp'] = False

    def __len__(self): return len(self._dlr)

    def __repr__(self):
        out = '-'*20 + ' Dyson solver ' + '-'*20
        out += '\nΛ = {} (Λ ∼ ωmax*β))'.format(self.lamb)
        out += '\nε = {:1.0e}'.format(self.eps)
        out += '\ndlr basis size = {}'.format(len(self))
        out += '\nscipy info: '
        out += '\n\tmethod = {}'.format(self.method)
        for key, val in self.options.items():
            out += '\n\t{}     =    {}'.format(key, val)
        out += '\n'+'-'*20 + '--------------' + '-'*20
        return out

    __str__ = __repr__

    # wrapping functions to the DLR class
    # abstract away the underlying DLR class functions

    @property
    def rank(self): return self._dlr.rank

    @property
    def dlr_nodes(self): return self._dlr.dlrrf

    def get_tau(self, beta): return self._dlr.get_tau(beta)
    def get_iom(self, beta): return self._dlr.get_matsubara_frequencies(beta)
    
    # DLR <-> Matsubara
    def dlr_from_iom(self, g_iwaa, beta): return self._dlr.dlr_from_matsubara(g_iwaa, beta)
    def iom_from_dlr(self, g_xaa,beta): return self._dlr.matsubara_from_dlr(g_xaa, beta)
    def fit_dlr_from_iom(self, mesh, g_iwaa, beta): return self._dlr.lstsq_dlr_from_matsubara(mesh, g_iwaa, beta)
    def eval_dlr_iom(self, g_xaa, mesh, beta): return self._dlr.eval_dlr_freq(g_xaa, mesh, beta)

    # DLR <-> tau
    def dlr_from_tau(self, g_xaa): return self._dlr.dlr_from_tau(g_xaa)
    def fit_dlr_from_tau(self, mesh, g_iaa, beta): return self._dlr.lstsq_dlr_from_tau(mesh, g_iaa, beta)
    def eval_dlr_tau(self, g_xaa, mesh, beta): return self._dlr.eval_dlr_tau(g_xaa, mesh, beta)


    # function to precompute Mkl
    def _compute_mkl(self):
        Mkl = np.zeros((len(self), len(self)), dtype=np.float128)
        for iwk, wk in enumerate(self.dlr_nodes):
            for iwl, wl in enumerate(self.dlr_nodes):
                K0wk, Kbwk = kernel(np.array([0.,1.]), np.array([wk]))
                K0wl, Kbwl = kernel(np.array([0.,1.]), np.array([wl]))
                if np.fabs(wk+wl) < 1e-13: 
                    Mkl[iwk,iwl] = K0wk*K0wl
                else: 
                    Mkl[iwk, iwl] = (K0wk*K0wl - Kbwk*Kbwl)
                    Mkl[iwk, iwl] /= (wk+wl)
        return Mkl

    def _solve_minimize_problem(self,
                                 beta,               # inverse temperature
                                 sigma_moments,      # high-freq moments of Σ
                                 g_taa = None,         # G(tau)  data
                                 g0_taa = None,        # G0(tau) data
                                 g_iwaa = None,        # G(iw)   data
                                 g0_iwaa = None,       # G0(iw)  data
                                 tau = None,           # tau mesh
                                 iw  = None,           # tau mesh
                                ):

        raise NotImplementedError

        tau_k = self.get_tau(beta) / beta 
        inu_k = self.get_iom(beta)
        wl = self.dlr_nodes

        Ktilde = -1*iw_kernel(inu_k, wl/beta)

        one = np.ones(len(self)).reshape(-1,1)

        Ktilde_mT1 = np.linalg.solve(Ktilde.T, one)
        Dtilde = np.diag((g0_iwaa*g_iwaa))

        Sigma1_tilde = one.T @ np.linalg.solve(Ktilde, (1/g0_iwaa - 1/g_iwaa) )

        constraint = Sigma_infty*(one.T@Ktilde_mT1) - Sigma_1 

        u = Ktilde.T @ (np.linalg.solve(Dtilde, Ktilde_mT1) )

        M1u = np.linalg.solve(self.Mkl, u)

        h = (g_iwaa - g0_iwaa)

        rho = h + ( (constraint - Sigma1_tilde ) / (u.T@M1u) ) * (Ktilde @ M1u)

        sigma = np.linalg.sovle(Dtilde, rho)




    # run the scipy minimization
    def _minimize_dyson_equation(self,
                                 beta,               # inverse temperature
                                 sigma_moments,      # high-freq moments of Σ
                                 g_taa = None,         # G(tau)  data
                                 g0_taa = None,        # G0(tau) data
                                 g_iwaa = None,        # G(iw)   data
                                 g0_iwaa = None,       # G0(iw)  data
                                 tau = None,           # tau mesh
                                 iw  = None,           # tau mesh
                                ):

        
        # fold and unfold complex numbers
        def merge_re_im(x):
            x_d, x_u = x[:nx*no], x[nx*no:]
            re, im = np.split(x_u, 2)
            x_u = re + 1.j * im
            return x_d, x_u

        def split_re_im(x_d, x_u):
            return np.concatenate((
                np.array(x_d.real, dtype=float),
                np.array(x_u.real, dtype=float),
                np.array(x_u.imag, dtype=float)))
                                       
        def sig_from_x(x):
            x_d, x_off = x[:2*nx*no], x[2*nx*no:]
            x_u, x_l = np.split(x_off, 2)
            sig = np.zeros((nx, no, no), dtype=dtype)
            for func, x in zip((sym.set_x_d, sym.set_x_u, sym.set_x_l), (x_d, x_u, x_l)):
                re, im = np.split(x,2)
                x = re + 1.j*im
                func(sig, x)
            return sig

        def x_from_sig(sig):
            x_d, x_u, x_l = sym.get_x_d(sig), sym.get_x_u(sig), sym.get_x_l(sig);
            x = np.concatenate((np.array(x_d.real, dtype=float),
                            np.array(x_d.imag, dtype=float),
                            np.array(x_u.real, dtype=float),
                            np.array(x_u.imag, dtype=float),
                            np.array(x_l.real, dtype=float),
                            np.array(x_l.imag, dtype=float)
                          ))
        
            return x
        
        def mat_vec(mat):
            v_d, v_u, v_l = sym.get_x_d(mat[None,...]), sym.get_x_u(mat[None,...]), sym.get_x_l(mat[None,...])
            return np.concatenate((
                np.array(v_d.real, dtype=float),
                np.array(v_d.imag, dtype=float),
                np.array(v_u.real, dtype=float),
                np.array(v_u.imag, dtype=float),
                np.array(v_l.real, dtype=float),
                np.array(v_l.imag, dtype=float)
            ))
            
        def constraint_func(x):
            # constraint condition: -∑σk =  Σ_1
            sig = self.dlr_from_iom(sig_from_x(x), beta)
            mat = -sig.sum(axis=0) 
            vec = mat_vec(mat)
            return vec
        

        # target function  
        def dyson_difference(x):
            sig = sig_from_x(x)
            sig_iwaa = sig + sig_infty
            #  G - G0 - G0*Σ*G = 0 done on the DLR nodes
            r_iwaa = g_iwaa - g0_iwaa - g0_iwaa@sig_iwaa@g_iwaa
            # compute DLR of rk_iwaa
            r_xaa = self.fit_dlr_from_iom(freq, r_iwaa, beta)
            # ||R||^2 = r^T @ M @ r
            R2 = np.einsum('mnk, kl, lnm->nm', r_xaa.T.conj(), self.Mkl, r_xaa).flatten()
            # the Frobeinus norm
            return np.sqrt(np.sum(R2)).real

        #assert g_iaa.shape[1:] == g0_iaa.shape[1:] == sigma_moments.shape[1:], "number of orbs inconsistent across G, G0, and moments"
        
        nx = len(self)
        if g_taa is not None:
            ni, no, _ = g_taa.shape
        elif g_iwaa is not None:
            ni, no, _ = g_iwaa.shape
        else:
            raise ValueError("Please provide either Green's function data on tau or iw")

        shape_xaa = (nx, no, no)
        N = (no*(no-1))//2
        dtype = complex

        # self energy <-> vector conversion
        sym = Symmetrizer(nx, no)

        # constraint
        sig_infty, sigma_1 = sigma_moments[0], sigma_moments[1]
        bound = mat_vec(sigma_1)
        constraints = (NonlinearConstraint(constraint_func, bound, bound))
            
        freq = self.get_iom(beta)
        
        # dlr fit to G and G0 
        if g_taa is not None:
            if tau is None: g_xaa = self.dlr_from_tau(g_taa)
            else: g_xaa = self.fit_dlr_from_tau(tau, g_taa, beta)

        if g0_taa is not None:
            if tau is None: g0_xaa = self.dlr_from_tau(g0_taa)
            else: g0_xaa = self.fit_dlr_from_tau(tau, g0_taa, beta)

        if g_iwaa is not None:
            if iw is None: g_xaa = self.dlr_from_iom(g_iwaa, beta)
            else: g_xaa = self.fit_dlr_from_iom(freq, g_iwaa, beta)

        if g0_iwaa is not None:
            if iw is None: g0_xaa = self.dlr_from_iom(g0_iwaa, beta)
            else: g0_xaa = self.fit_dlr_from_iom(iw, g0_iwaa, beta)

        if self.verbose:
            eval_tau = tau if tau is not None else self.get_tau(beta) #np.linspace(0, beta, len(g_iaa))
            g, g0 = self.eval_dlr_tau(g_xaa, eval_tau, beta), self.eval_dlr_tau(g0_xaa, eval_tau,  beta)
            print('initial DLR fits to G(τ) and G0(τ)')
            print(f'max|G(τ) - Gdlr(τ)| = {np.max(np.abs(g-g_iaa)):.6e}')
            print(f'max|G0(τ) - G0dlr(τ)| = {np.max(np.abs(g0-g0_iaa)):.6e}')

        # compute and obtain initial Σ
        g_iwaa, g0_iwaa = self.iom_from_dlr(g_xaa, beta), self.iom_from_dlr(g0_xaa, beta)

        # the DLR representable part of the self-energy
        # initial guess for DLR Sigma only use of Dyson equation!
        sig0_iwaa = np.linalg.inv(g0_iwaa)-np.linalg.inv(g_iwaa)-sig_infty
       
        # optimize Σ(iν)
        x_init = x_from_sig(sig0_iwaa)
        
        cb = Callback(eval_func= lambda x : sig_from_x(x)+sig_infty, diff_func=dyson_difference, method=self.method)

        solution = minimize(dyson_difference, 
                            x_init,
                            method=self.method,
                            constraints=constraints,
                            options=self.options,
                            callback = cb
                           )
        
        if self.verbose: print(solution.message)
        if not solution.success: print('[WARNING] Minimization did not converge! Please proceed with caution!')
        
        sig_iwaa = sig_from_x(solution.x)
        sig_xaa = self.dlr_from_iom(sig_iwaa, beta)
            
        if self.verbose: print(f'Σ1 constraint diff: {np.max(np.abs(-sig_xaa.sum(axis=0)-sigma_1)):.4e}')

        result = MinimizerResult(solution,
                                  sig_xaa,
                                  g_xaa,
                                  g0_xaa,
                                  cb.history
                                 )
        return result

    # main solve function
    def solve(self, Sigma_iw = None, 
                    G_tau = None, 
                    G0_tau = None, 
                    G_iw = None, 
                    G0_iw = None, 
                    Sigma_moments = None, 
                    beta = None, 
                    om_mesh = None, 
                    tau_mesh = None
                    ):

        result = None

        assert G_tau is not None or G_iw is not None, "Please provide G"
        assert G_tau is None or G_iw is None, "Please only provide one G in tau or matsubara"
        assert G0_tau is not None or G0_iw is not None, "Please provide G0"
        assert G0_tau is None and G0_iw is None, "Please only provide one G0 in tau or matsubara"

        if G_tau is not None and G_iw is None:   
            G = G_tau
            G_tau_or_freq = 'tau'
        elif G_tau is None and G_iw is not None: 
            G = G_iw
            G_tau_or_freq = 'freq'

        if G0_tau is not None and G0_iw is None:   
            G0 = G0_tau
            G0_tau_or_freq = 'tau'
        elif G0_tau is None and G0_iw is not None: 
            G0 = G0_iw
            G0_tau_or_freq = 'freq'

        # we are working with a TRIQS Green's function/ Block Green's function object
        if all(list(map(is_block_gf, [G, G0]))):

            beta = G.mesh.beta
            if G_tau_or_freq == 'tau':
                tau  = np.array([float(x) for x in G.mesh]) if tau_mesh is None else tau_mesh
            if G_tau_or_freq == 'freq':
                iw  = np.array([complex(x) for x in G.mesh]) if om_mesh is None else om_mesh

            if Sigma_iw is not None:
                Sigma_iw_fit = Sigma_iw.copy()
                om_mesh = np.array([complex(x) for x in Sigma_iw_fit.mesh])

            dlr_results = {}
            for block, sig in Sigma_iw_fit:

                dlr_results[block] =  self._minimize_dyson_equation(beta, 
                                                                    Sigma_moments[block],
                                                                    g_taa = G[block].data if G_tau_or_freq == 'tau' else None,
                                                                    g_iwaa = G[block].data if G_tau_or_freq == 'freq' else None,
                                                                    g0_taa = G0[block].data if G0_tau_or_freq == 'tau' else None,
                                                                    g0_iwaa = G0[block].data if G0_tau_or_freq == 'freq' else None,
                                                                    tau=tau,
                                                                    iw=iw,
                                                              )
                # DLR representable part of self energy
                Sigma_iw_fit[block].data[:] = self.eval_dlr_iom(dlr_results[block].sig_xaa, om_mesh, beta)
                # add constant back to obtain true self energy
                Sigma_iw_fit[block].data[:] +=  Sigma_moments[block][0]

        # our Green's functions are just numpy arrays
        elif all(list(map(is_array, [G, G0]))):

            assert beta is not None, "must provide a beta!"

            dlr_results = self._minimize_dyson_equation(beta,
                                                        Sigma_moments,
                                                        g_taa = G    if G_tau_or_freq == 'tau' else None,
                                                        g_iwaa = G   if G_tau_or_freq == 'freq' else None,
                                                        g0_taa = G0  if G0_tau_or_freq == 'tau' else None,
                                                        g0_iwaa = G0 if G0_tau_or_freq == 'freq' else None,
                                                        tau=tau_mesh,
                                                        iw=om_mesh
                                                        ) 

            if om_mesh is None:
                Sigma_iw_fit = self.iom_from_dlr(dlr_results.sig_xaa, beta)
                Sigma_iw_fit += Sigma_moments[0]
            else:
                Sigma_iw_fit = self.eval_dlr_iom(dlr_results.sig_xaa, om_mesh, beta) 
                Sigma_iw_fit += Sigma_moments[0]

        else: raise ValueError

        result = SolverResult(Sigma_iw_fit,
                               G,
                               G0,
                               Sigma_moments,
                               dlr_results
                        )

        return result
