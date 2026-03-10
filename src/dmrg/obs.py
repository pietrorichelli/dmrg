import os
import shutil
import numpy as np

class observables():
    """
    Observable computation on Matrix Product States.
    Calculates single-site expectation values, two-point correlations,
    entanglement entropy, and correlation functions from optimized MPS.
    
    Attributes:
        mps : MPS
            Matrix Product State for computing observables
        L : int
            Number of sites in the chain
        d : int
            Physical dimension
    
    Methods:
        __init__(MPS):
            Initialize with MPS object
        single_site(site, obs):
            Compute single-site expectation value <obs> at given site
        bound_left(site, obs):
            Compute observable with boundary at left edge up to site
        bound_right(site, obs):
            Compute observable with boundary at right edge down to site
        all_corr(path, site, string, obs1, obs2):
            Compute all correlation functions from site to right boundary, write to file
        two_sites(site1, site2, string, obs1, obs2):
            Compute two-point correlation <obs1(site1) string obs2(site2)>
        left_EE():
            Compute entanglement entropy at left-center bond
        right_EE():
            Compute entanglement entropy at right-center bond
    """

    # Initialize observable calculator with MPS
    def __init__(self,MPS):
        self.mps = MPS
        self.L = MPS.L 
        self.d = MPS.d

    # Compute single-site expectation value <obs> at given site
    def single_site(self,site,obs):
        ten = np.tensordot(self.mps.read(site),self.mps.readS(site),(2,0)) 
        return np.tensordot(np.tensordot(obs,ten,(0,0)),np.conj(ten),((0,1,2),(0,1,2)))

    # Compute observable contraction with boundary tensor at left, result split into two parts
    def bound_left(self,site,obs):
        tenS = np.tensordot(self.mps.read(site),self.mps.readS(site),(2,0)) 

        ob1 = np.tensordot(np.tensordot(self.mps.read(0),obs,(0,0)),np.conj(self.mps.read(0)),(0,0))
        ob1 = np.tensordot(ob1,tenS,(0,1))
        ob1 = np.tensordot(ob1,np.conj(tenS),((0,1),(1,0)))
        ob1 = np.trace(ob1)
        ob2 = np.tensordot(np.tensordot(obs,tenS,(0,0)),np.conj(tenS),((0,1,2),(0,1,2)))

        return ob1,ob2

    # Compute observable contraction with boundary tensor at right, result split into two parts
    def bound_right(self,site,obs):
        tenS = np.tensordot(self.mps.read(site),self.mps.readS(site-1),(1,0)) 

        ob1 = np.tensordot(np.tensordot(obs,tenS,(0,0)),np.conj(tenS),((0,1,2),(0,1,2)))
        ob2 = np.tensordot(np.tensordot(self.mps.read(self.L-1),obs,(0,0)),np.conj(self.mps.read(self.L-1)),(1,0))
        ob2 = np.tensordot(ob2,tenS,(0,1))
        ob2 = np.tensordot(ob2,np.conj(tenS),((0,1,2),(1,0,2)))
        
        return ob1,ob2
    
    # Compute all correlation functions from site to right boundary using string operator, append to file
    def all_corr(self,path,site,string,obs1,obs2=None):
        if obs2 is None:
            obs2 = obs1
        ten = np.tensordot(self.mps.read(site),self.mps.readS(site),(2,0)) 
        
        cont1 = np.tensordot(np.tensordot(obs1,ten,(0,0)),np.conj(ten),((0,1),(0,1)))
        
        
        for i in range(site+1,self.L-1):
            cont2 = np.tensordot(np.tensordot(obs2,self.mps.read(i),(0,0)),np.conj(self.mps.read(i)),((0,2),(0,2)))
            if i > site + 1:
                cont1 = np.tensordot(cont1,np.tensordot(string,self.mps.read(i-1),(0,0)),(0,1))
                cont1 = np.tensordot(cont1,np.conj(self.mps.read(i-1)),((0,1),(1,0))) 
            
            res = np.tensordot(cont1,cont2,((0,1),(0,1)))
        
            with open(path,'a') as f:
                f.write(f'{site} {i} {res}\n')

    # Compute two-point correlation <obs1(site1) string obs2(site2)> between distant sites
    def two_sites(self,site1,site2,string,obs1,obs2=None):
        if obs2 is None:
            obs2 = obs1

        ten = np.tensordot(self.mps.read(site1),self.mps.readS(site1),(2,0))
        cont1 = np.tensordot(np.tensordot(obs1,ten,(0,0)),np.conj(ten),((0,1),(0,1)))
        cont2 = np.tensordot(np.tensordot(obs2,self.mps.read(site2),(0,0)),np.conj(self.mps.read(site2)),((0,2),(0,2)))
        
        for i in range(site1+1,site2):
            cont1 = np.tensordot(cont1,np.tensordot(string,self.mps.read(i),(0,0)),(0,1))
            cont1 = np.tensordot(cont1,np.conj(self.mps.read(i)),((0,1),(1,0))) 

        return np.tensordot(cont1,cont2,((0,1),(0,1)))

    # Compute entanglement entropy at left-center bond (between sites 1 and 2)
    def left_EE(self):
        d0,d1,d2 = self.mps.read(2).shape
        ten1 = np.einsum('ij,kli->lkj',self.mps.readS(2),self.mps.read(2))
        mat1 = np.reshape(ten1,(d1,d0*d2))
        l1,c1,r1 = np.linalg.svd(mat1,full_matrices=False)

        d0,d1,d2 = self.mps.read(1).shape
        # ten0 = np.tensordot(l1@np.diag(c1),self.mps.read(1),(0,2))
        ten0 = np.einsum('ij,kli->lkj',l1@np.diag(c1),self.mps.read(1))
        mat0 = np.reshape(ten0,(d1,d0*d2))
        l0,c0,r0 = np.linalg.svd(mat0,full_matrices=False)

        return -c0**2@np.log(c0**2),-c1**2@np.log(c1**2)

    # Compute entanglement entropy at right-center bond (between sites L-3 and L-2)
    def right_EE(self):
        d0,d1,d2 = self.mps.read(self.L-2).shape
        ten = np.tensordot(self.mps.readS(self.L-3),self.mps.read(self.L-2),(0,1))
        mat = np.reshape(ten,(d0*d1,d2))
        l,c,r = np.linalg.svd(mat,full_matrices=False)
        return -c**2@np.log(c**2)