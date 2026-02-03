import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import ArpackNoConvergence

class EffH():

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

    def __str__(self):
        return 'Effective Hamiltonian on the Cayley tree'
    
    def __repr__(self):
        return 'Effective Hamiltonian on the Cayley tree'

    def matvec(self,psi):
        h = self.H
        psi = np.reshape(psi,(self.c1,self.d,self.d,self.c2))
        
        x = np.tensordot(self.L,h.mpo(p=self.site),(1,2))
        x = np.tensordot(x,psi,[(0,2),(0,1)])
        x = np.tensordot(x,h.mpo(p=self.site+1),[(2,3),(2,0)])
        x = np.tensordot(x,self.R,[(4,2),(1,0)])
        
        return np.reshape(x,(self.c1*self.d*self.d*self.c2))
    
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
            psi = self.matvec(psi) - beta*vecs[-2]
            alpha = psi.conj()@vecs[-1]
            psi = psi - alpha * vecs[-1] 
            T[i, i] = alpha.real
            T[i-1, i] = T[i, i-1] = beta    
        
        return T, (np.array(vecs).T).conj()

    def lanc_iter_old(self,psi0,exc='off'):
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

    def lanczos_grd(self,psi0=None,exc='off'):
        if psi0 is None:
            psi0 = np.random.rand(self.c1*self.c2*self.d**2)
        T, vecs = self.lanc_iter_old(psi0,exc=exc)

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