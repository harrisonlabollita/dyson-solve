import numpy as np
from pydlr import kernel, dlr
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

from triqs.gf import *

is_block_gf = lambda x : isinstance(x, BlockGf)
is_array = lambda x : isinstance(x, np.ndarray)

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


class Result(dict):
    def __getattr__(self, name):
        try: 
            return self[name]
        except KeyError as e: raise AttributeError(name) from e
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Dyson:

    def __init__(self, lamb, 
                       eps=1e-9,                
                       method='trust-constr', 
                       options=dict(maxiter=10000),
                       verbose=True,
                       **kwargs
                       ):

        self.lamb = lamb                              # dlr lambda
        self.eps  = eps                               # dlr epsilon
        self.d    = dlr(lamb=self.lamb, eps=self.eps, # dlr class
                       **kwargs)
        self.Mkl  = self._compute_mkl()                # compute residual matrix
        self.method = method                          # scipy minimize method
        self.options = options                        # minimize options
        self.verbose = verbose

        # if verbose add display to minimize options
        if self.verbose: self.options['disp'] = True
        else: self.options['disp'] = False

    def __len__(self): return len(self.d)

    def __repr__(self):
        out = '-'*20 + ' Dyson solver ' + '-'*20
        out += '\nΛ = {} (Λ ∼ ωmax*β))'.format(self.lamb)
        out += '\nε = {:1.0e}'.format(self.eps)
        out += '\ndlr basis size = {}'.format(len(self))
        out += '\nscipy info: '
        out += '\n\tmethod = {}'.format(self.method)
        for key, val in self.options.items():
            out += '\n\t{}     =    {}'.format(key, val)
        return out

    __str__ = __repr__


    # function to precompute Mkl
    def _compute_mkl(self):
        Mkl = np.zeros((len(self.d), len(self.d)), dtype=np.float128)
        for iwk, wk in enumerate(self.d.dlrrf):
            for iwl, wl in enumerate(self.d.dlrrf):
                K0wk, Kbwk = kernel(np.array([0.,1.]), np.array([wk]))
                K0wl, Kbwl = kernel(np.array([0.,1.]), np.array([wl]))
                if np.fabs(wk+wl) < 1e-13:
                    Mkl[iwk,iwl] = K0wk*K0wl
                else:
                    Mkl[iwk, iwl] = (K0wk*K0wl - Kbwk*Kbwl)
                    Mkl[iwk, iwl] /= ((wk+wl))
        return Mkl


    def _constrained_lstsq_dlr_from_tau(self,
                                       g_iaa,         # G data
                                       g0_iaa,        # G0 data
                                       beta,          # inverse temperature
                                       sigma_moments, # high-freq moments of Σ
                                       tau=None,      # tau mesh
                                      ):

        assert g_iaa.shape[1:] == g0_iaa.shape[1:] == sigma_moments.shape[1:], "number of orbs inconsistent across G, G0, and moments"
        
        nx = len(self.d)
        ni, no, _ = g_iaa.shape
        shape_xaa = (nx, no, no)
        N = (no*(no-1))//2

        dtype = complex
        nX = nx * (no + 2*N)
        
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
                                       
        # Self-energy <-> vector conversion
        sym = Symmetrizer(nx, no)

        def sig_from_x(x):
            x_d, x_off = x[:2*nx*no], x[2*nx*no:]
            x_u, x_l = np.split(x_off, 2)
        
            re, im = np.split(x_d, 2)
            x_d = re + 1.j * im
            
            re, im = np.split(x_u, 2)
            x_u = re + 1.j * im
            
            re, im = np.split(x_l, 2)
            x_l = re + 1.j * im
            
            sig = np.zeros((nx, no, no), dtype=dtype)
            sym.set_x_u(sig, x_u)
            sym.set_x_l(sig, x_l)
            sym.set_x_d(sig, x_d)
            
            return sig

        def x_from_sig(sig):
            x_d = sym.get_x_d(sig);
            x_u = sym.get_x_u(sig);
            x_l = sym.get_x_l(sig);
            x = np.concatenate((np.array(x_d.real, dtype=float),
                            np.array(x_d.imag, dtype=float),
                            np.array(x_u.real, dtype=float),
                            np.array(x_u.imag, dtype=float),
                            np.array(x_l.real, dtype=float),
                            np.array(x_l.imag, dtype=float)
                          ))
        
            return x
        
        
        # constraint
        sig_infty, sigma_1 = sigma_moments[0], sigma_moments[1]
            
        def mat_vec(mat):
            v_d = sym.get_x_d(mat[None, ...])
            v_u = sym.get_x_u(mat[None, ...])
            v_l = sym.get_x_l(mat[None, ...])
            
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
            sig = self.d.dlr_from_matsubara(sig_from_x(x), beta)
            mat = -sig.sum(axis=0) 
            vec = mat_vec(mat)
            return vec
        
        bound = mat_vec(sigma_1)
        
        constraints = (NonlinearConstraint(constraint_func,
                                               bound, bound))

        # target function  
        def dyson_difference(x):
            
            sig = sig_from_x(x)
            sig_iwaa = sig + sig_infty
            #  G - G0 - G0*Σ*G = 0 done on the DLR nodes
            r_iwaa = g_iwaa - g0_iwaa - g0_iwaa@sig_iwaa@g_iwaa
            
            # compute DLR of rk_iwaa
            r_xaa = self.d.lstsq_dlr_from_matsubara(freq, r_iwaa, beta)

            # ||R||^2 = r^T @ M @ r
            R2 = np.einsum('mnk, kl, lnm->nm', r_xaa.T.conj(), self.Mkl, r_xaa).flatten()
            
            # the Frobeinus norm
            return np.sqrt(np.sum(R2)).real
            
        freq = self.d.get_matsubara_frequencies(beta)
        
        # dlr fit to G and G0 
        if tau is None:
            g_xaa = self.d.dlr_from_tau(g_iaa)
            g0_xaa = self.d.dlr_from_tau(g0_iaa)
        else:
            g_xaa = self.d.lstsq_dlr_from_tau(tau, g_iaa, beta)
            g0_xaa = self.d.lstsq_dlr_from_tau(tau, g0_iaa, beta)
        
        if self.verbose:
            if tau is None:
                g=self.d.tau_from_dlr(g_xaa)
                g0=self.d.tau_from_dlr(g0_xaa)
            else:
                g=self.d.eval_dlr_tau(g_xaa, tau, beta)
                g0=self.d.eval_dlr_tau(g0_xaa, tau,  beta)
            
            print('initial DLR fits to G(τ) and G0(τ)')
            print(f'|G(τ) - Gdlr(τ)| = {np.max(np.abs(g-g_iaa)):.6e}')
            print(f'|G0(τ) - G0dlr(τ)| = {np.max(np.abs(g0-g0_iaa)):.6e}')
        
        # compute and obtain initial Σ
        g_iwaa = self.d.matsubara_from_dlr(g_xaa, beta)
        g0_iwaa = self.d.matsubara_from_dlr(g0_xaa, beta)
        
        # the DLR representable part of the self-energy
        sig0_iwaa = np.linalg.inv(g0_iwaa)-np.linalg.inv(g_iwaa)-sig_infty
        
        if self.verbose:
            check = sig_from_x(x_from_sig(sig0_iwaa))
            assert np.allclose(check, sig0_iwaa), "sigma converter is broken!"
        
        # optimize Σ(iν)
        x_init = x_from_sig(sig0_iwaa)
        
        history=[]
        def callback(x, status):
            sig=sig_from_x(x)
            sig_iwaa = sig+sig_infty
            history.append((x,sig_iwaa, dyson_difference(x)))

        solution = minimize(dyson_difference, 
                       x_init,
                       method=self.method,
                       constraints=constraints,
                       options=self.options,
                       callback=callback if self.method == 'trust-constr' else lambda x : callback(x, None)
            )
        
        if self.verbose: print(solution.success, solution.message)
        if not solution.success: print('[WARNING] Minimization did not converge!')
        
        sig_iwaa = sig_from_x(solution.x)
        sig_xaa = self.d.dlr_from_matsubara(sig_iwaa, beta)
            
        if self.verbose: print(f'Σ1 constraint diff: {np.max(np.abs(-sig_xaa.sum(axis=0)-sigma_1)):.4e}')

        result = Result(solution = solution,
                        sig_xaa  = sig_xaa,
                        g_xaa    = g_xaa,
                        g0_xaa   = g0_xaa,
                        callback  = history
                        )
        return result

    def solve(self, Sigma_iw=None, 
                    G_tau=None, 
                    G0_tau=None, 
                    Sigma_moments=None, 
                    beta=None, 
                    om_mesh=None, 
                    tau_mesh=None):

        result = None
        
        # we are working with a TRIQS Green's function/ Block Green's function object
        if all(list(map(is_block_gf, [G_tau, G0_tau]))):

            beta = G_tau.mesh.beta
            tau  = np.array([float(x) for x in G_tau.mesh]) if tau_mesh is None else tau_mesh

            #TODO: Sigma_iw shouldn't be required for BlockGf option
            Sigma_iw_fit = Sigma_iw.copy()
            iw = np.array([complex(x) for x in Sigma_iw_fit.mesh])

            dlr_results = {}

            for block, sig in Sigma_iw_fit:

                dlr_results[block] =  self._constrained_lstsq_dlr_from_tau(G_tau[block].data,
                                                                           G0_tau[block].data,
                                                                           beta,
                                                                           Sigma_moments[block],
                                                                           tau=tau
                                                              )

                Sigma_iw_fit[block].data[:] = self.d.eval_dlr_freq(dlr_results[block].sig_xaa, iw, beta)
                Sigma_iw_fit[block].data[:] +=  Sigma_moments[block][0]


        # our Green's functions are just numpy arrays
        elif all(list(map(is_array, [G_tau, G0_tau]))):

            assert beta is not None, "must provide a beta!"

            dlr_results = self._constrained_lstsq_dlr_from_tau(G_tau,
                                                              G0_tau,
                                                              beta,
                                                              Sigma_moments,
                                                              tau=tau_mesh
                                                              )

            if om_mesh is None:
                Sigma_iw_fit = self.d.matsubara_from_dlr(dlr_results.sig_xaa)
                Sigma_iw_fit += Sigma_moments[0]
            else:
                Sigma_iw_fit = self.d.eval_dlr_freq(dlr_results.sig_xaa,om_mesh,beta)
                Sigma_iw_fit += Sigma_moments[0]

        else:
            raise ValueError

        result = Result(Sigma_iw      = Sigma_iw_fit,
                        G0_tau        = G0_tau,
                        G_tau         = G_tau,
                        Sigma_moments = Sigma_moments,
                        dlr_optim     = dlr_results
                        )

        return result
