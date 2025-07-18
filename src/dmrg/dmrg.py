import numpy as np

from .MPS import MPS
from .cont import CONT
from .lanczos import EffH

class dmrg():
    """
        Class that runs the DMRG algorithm on a 1 dimensional system

            Attributes:
                - cont: Class DMRG.contractions
    """

    def __init__(self,cont,chi=100,cut=1e-12):
        self.cont = cont
        self.mps = cont.mps
        self.chi = chi
        self.h = cont.h
        self.L = cont.L
        self.count = cont.count
        self.cut = cut
        self.d = self.mps.d

    
    def infinite(self):

        En = np.zeros(self.L//2-1)
        
        env_left = self.cont.left(0)
        env_right = self.cont.right(self.L-1)

        for i in range(1,int(self.mps.L/2)):

            H = EffH(env_left,env_right,self.h)
            En[i-1],grd = H.lanczos_grd()

            mat = np.reshape(grd,(H.c1*H.d,H.d*H.c2))
            l,c,r = np.linalg.svd(mat)

            bound = min(len(c[c>self.cut]),self.chi)
           
            l = l[:,:bound]
            c = c[:bound]
            r = r[:bound,:]

            self.mps.write_left(i,l)
            self.mps.write_right(self.mps.L-i-1,r)
            
            env_left = self.cont.add(i,'l')
            env_right = self.cont.add(self.mps.L-i-1,'r')
            
        
        self.mps.writeS(i,np.diag(c))
        if i > 0:
            self.mps.delete(i-1)
            
        return En

    def step2sites(self,site,dir,exc='off',stage=None):

        env_left,env_right = self.cont.env_prep(site)
    
        H = EffH(env_left,env_right,self.h)

        if dir == 'l' or dir == 'bl':
            init_vec = np.tensordot(self.mps.read(site),np.tensordot(self.mps.read(site+1),self.mps.readS(site+1),(2,0)),(2,1))
            init_vec = dmrg.remish(init_vec)
        if dir == 'r' or dir == 'br':
            init_vec = np.tensordot(np.tensordot(self.mps.readS(site-1),self.mps.read(site),(0,1)),self.mps.read(site+1),(2,1))

        init_vec = np.reshape(init_vec,np.prod(init_vec.shape))

        # En_pre = np.conj(init_vec)@H.matvec(init_vec)
        
        if stage == None:
            En,grd = H.lanczos_grd(psi0=None,exc=exc)
            grd_state = 1/np.sqrt(np.conj(grd)@grd)*grd
        if stage == 'Final':
            grd_state = 1/np.sqrt(init_vec@np.conj(init_vec))*init_vec
            En = np.conj(grd_state)@H.matvec(grd_state)
        
        grd_state = np.reshape(grd_state,(H.c1*H.d,H.d*H.c2))

        l,c,r = np.linalg.svd(grd_state,full_matrices=False)
        
        bound = min(len(c[c>self.cut]),self.chi)
        l = l[:,:bound]
        c = c[:bound]
        r = r[:bound,:]
        
        self.mps.write_left(site,l)
        self.mps.write_right(site+1,r)
        self.mps.writeS(site,np.diag(c))

        if dir == 'r':
            self.cont.add(site,'l')
            if site == self.L -3:
                self.cont.add(site+1,'r')
        
        if dir == 'l':
            self.cont.add(site+1,'r')
            if site == 1:
                self.cont.add(site,'l')
        
        return En, -c**2@np.log(c**2)


    def remish(ten):
        d0,d1,d2,d3 = ten.shape
        res = np.zeros((d1,d0,d2,d3),dtype='complex')
        for i0 in range(d0):
            for i1 in range(d1):
                res[i1,i0,:,:] = ten[i0,i1,:,:]

        return res
    