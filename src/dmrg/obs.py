import os
import shutil
import numpy as np

class observables():

    def __init__(self,MPS):
        self.mps = MPS
        self.L = MPS.L 

    def single_site(self,site,obs):
        ten = np.tensordot(self.mps.read(site),self.mps.readS(site),(2,0)) 
        np.tensordot(obs,ten,(0,0))
        return np.tensordot(np.tensordot(obs,ten,(0,0)),np.conj(ten),((0,1,2),(0,1,2)))

    def all_corr(self,path,site,obs1,obs2=None):
        if obs2 == None:
            obs2 = obs1
        ten = np.tensordot(self.mps.read(site),self.mps.readS(site),(2,0)) 
        
        cont1 = np.tensordot(np.tensordot(obs1,ten,(0,0)),np.conj(ten),((0,1),(0,1)))
        
        for i in range(site+1,self.L-1):
            cont2 = np.tensordot(np.tensordot(obs2,self.mps.read(i),(0,0)),np.conj(self.mps.read(i)),((0,2),(0,2)))
            if i > site + 1:
                cont1 = np.tensordot(cont1,self.mps.read(i-1),(0,1))
                cont1 = np.tensordot(cont1,np.conj(self.mps.read(i-1)),((0,1),(1,0))) 
            
            res = np.tensordot(cont1,cont2,((0,1),(0,1)))
        
            with open(path,'a') as f:
                f.write(f'{site} {i} {res}\n')

