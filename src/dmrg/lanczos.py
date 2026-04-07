import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import ArpackNoConvergence
from .OptimizedTensorContractor import contract

class EffH():
    """
    Effective Hamiltonian for two-site DMRG updates.
    Builds reduced Hamiltonian from left/right environments and MPO tensors,
    enabling efficient ground state and excited state optimization via Lanczos iterations.
    
    Attributes:
        L : np.ndarray
            Left environment tensor (shape: c1 x D x D)
        R : np.ndarray
            Right environment tensor (shape: c2 x D x D)
        H : MPO
            Matrix Product Operator (Hamiltonian) for the system
        site : int
            Left site index for two-site operation
        k : int
            Krylov subspace dimension for Lanczos iterations
        d : int
            Physical dimension from MPO
        c1 : int
            Left environment bond dimension
        c2 : int
            Right environment bond dimension
        len_vec : int
            Total vector length (c1 * c2 * d^2) for two-site state
        OTC : OptimizedTensorContractor
            Optimized tensor contraction engine for matvec operations
    
    Methods:
        __init__(L, R, H, site, k):
            Initialize effective Hamiltonian with environments and MPO
        __str__():
            Return string representation
        __repr__():
            Return representation string
        matvec(psi):
            Matrix-vector product: apply effective Hamiltonian to state vector
        lanc_iter(psi0, exc):
            Run Lanczos iterations building tridiagonal matrix and Krylov basis
        lanczos_grd(psi0, exc):
            Find ground (or excited) state via Lanczos eigenvalue solver
    """

    # Initialize effective Hamiltonian with left/right environments, MPO, and Lanczos parameters
    def __init__(self,L,R,H,site,k=300):
        self.L = L 
        self.R = R 
        self.H = H 
        self.k = k
        self.d = H.d
        self.site = site
        c1,_,_ = L.shape
        c2,_,_ = R.shape
        self.c1 = c1
        self.c2 = c2
        self.len_vec = c1*c2*self.d**2
        self.k = min(c1*c2*self.d**2,k)

    # Return string description of effective Hamiltonian
    def __str__(self):
        return 'Effective Hamiltonian on the Cayley tree'
    
    # Return representation string for effective Hamiltonian
    def __repr__(self):
        return 'Effective Hamiltonian on the Cayley tree'

    # Apply effective Hamiltonian to state vector: psi -> H|psi> by contracting environments with MPO
    def matvec(self,psi):
        h = self.H
        psi = np.reshape(psi,(self.c1,self.d,self.d,self.c2))
        
        x = contract('abc,dfbk,adgi,ghkl,ilm->cfhm',*(self.L,h.mpo(self.site),psi,h.mpo(self.site+1),self.R))
        
        return np.reshape(x,(self.c1*self.d*self.d*self.c2))

    # Run Lanczos iterations building tridiagonal matrix T and Krylov basis vectors
    def lanc_iter(self,psi0,exc='off'):
        psi0 = psi0/np.linalg.norm(psi0)
        vecs = [psi0] 

        T = np.zeros((self.k,self.k))

        psi = self.matvec(psi0)
        alpha = T[0, 0] = np.inner(psi0.conj(),psi).real
        psi = psi - alpha* vecs[-1]
        
        for i in range(1,self.k):
            beta = np.linalg.norm(psi)
            if beta  < 1e-8:
                T = T[:i, :i]
                break
            if exc == 'off':
                psi /= beta
            if exc == 'on':
                psi_o = psi
                for j in range(i):
                    psi_o -= (psi.conj()@vecs[j])*vecs[j]
                psi = psi_o/np.linalg.norm(psi_o)
            vecs.append(psi)
            psi = self.matvec(psi) 
            alpha = np.inner(vecs[-1].conj(), psi).real
            psi = psi - alpha * vecs[-1] - beta * vecs[-2]
            T[i, i] = alpha
            T[i-1, i] = T[i, i-1] = beta    
        return T, np.array(vecs).T

    # Find ground or excited state eigenvector by diagonalizing Lanczos tridiagonal matrix
    def lanczos_grd(self,psi0=None,exc='off'):
        if psi0 is None:
            psi0 = np.random.rand(self.c1*self.c2*self.d**2)
        T, vecs = self.lanc_iter(psi0,exc=exc)

        try:
            E,v = eigh_tridiagonal(np.diag(T),np.diag(T,k=1),select='i',select_range=(0,2))
            result = vecs @ v[:, np.argmin(E)]
            if exc == 'off':
                E = min(E)
                
        except ValueError as err:
            E,v = np.linalg.eigh(T)
            result = vecs @ v[:, np.argmin(E)]
            if exc == 'off':
                E = min(E)

        except ArpackNoConvergence as err:
            print('Lanczos did not converge !!!')
            E = psi0.conj()@self.matvec(psi0)
            result = psi0

        return E, result
