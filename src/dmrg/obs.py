import os
import shutil
import numpy as np

class observables():

    def __init__(self,MPS):
        self.mps = MPS
        self.L = MPS.L 
        self.d = MPS.d

    def single_site(self,site,obs):
        ten = np.tensordot(self.mps.read(site),self.mps.readS(site),(2,0)) 
        return np.tensordot(np.tensordot(obs,ten,(0,0)),np.conj(ten),((0,1,2),(0,1,2)))

    def bound_left(self,site,obs):
        tenS = np.tensordot(self.mps.read(site),self.mps.readS(site),(2,0)) 

        ob1 = np.tensordot(np.tensordot(self.mps.read(0),obs,(0,0)),np.conj(self.mps.read(0)),(0,0))
        ob1 = np.tensordot(ob1,tenS,(0,1))
        ob1 = np.tensordot(ob1,np.conj(tenS),((0,1),(1,0)))
        ob1 = np.trace(ob1)
        ob2 = np.tensordot(np.tensordot(obs,tenS,(0,0)),np.conj(tenS),((0,1,2),(0,1,2)))

        return ob1,ob2

    def bound_right(self,site,obs):
        tenS = np.tensordot(self.mps.read(site),self.mps.readS(site-1),(1,0)) 

        ob1 = np.tensordot(np.tensordot(obs,tenS,(0,0)),np.conj(tenS),((0,1,2),(0,1,2)))
        ob2 = np.tensordot(np.tensordot(self.mps.read(self.L-1),obs,(0,0)),np.conj(self.mps.read(self.L-1)),(1,0))
        ob2 = np.tensordot(ob2,tenS,(0,1))
        ob2 = np.tensordot(ob2,np.conj(tenS),((0,1,2),(1,0,2)))
        
        return ob1,ob2
    
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

    def right_EE(self):
        d0,d1,d2 = self.mps.read(self.L-2).shape
        ten = np.tensordot(self.mps.readS(self.L-3),self.mps.read(self.L-2),(0,1))
        mat = np.reshape(ten,(d0*d1,d2))
        l,c,r = np.linalg.svd(mat,full_matrices=False)
        return -c**2@np.log(c**2)