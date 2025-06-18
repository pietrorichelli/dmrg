import numpy as np

from .MPS import MPS
from .cont import contractions
from .lanczos import EffH

class dmrg():
    """
        Class that runs the DMRG algorithm on a 1 dimensional system

            Attributes:
                - cont: Class DMRG.contractions
    """

    def __init__(self,cont,chi=10):
        self.cont = cont
        self.mps = cont.mps
        self.chi = chi
        self.h = cont.h
        self.L = cont.L
        self.count = cont.count

    
    def infinite(self):

        # mps.write_bound(np.identity(2))

        En = np.zeros(self.L//2-1)
        
        env_left = self.cont.left(0)
        env_right = self.cont.right(self.L-1)

        for i in range(1,int(self.mps.L/2)):

            H = EffH(env_left,env_right,self.h)
            En[i-1],grd = H.lanczos_grd()

            mat = np.reshape(grd,(H.c1*H.d,H.d*H.c2))
            l,c,r = np.linalg.svd(mat)
            if len(c) > self.chi:
                l = l[:,:self.chi]
                c = c[:self.chi]
                r = r[:self.chi,:]

            ten_l = self.mps.left_ten(l)
            ten_r = self.mps.right_ten(r)

            self.mps.write(i,ten_l)
            self.mps.write(self.mps.L-i-1,ten_r)

            env_left = self.cont.add(i,'l')
            env_right = self.cont.add(self.mps.L-i-1,'r')
        
        self.mps.writeS(i,np.diag(c))
            
        return En

    def step2sites(self,site,dir,exc='off',stage=None):

        env_left,env_right = self.cont.env_prep(site)
    
        H = EffH(env_left,env_right,self.h)

        if dir == 'l':
            init_vec = np.tensordot(self.mps.read(site),np.tensordot(self.mps.read(site+1),self.mps.readS(site+1),(2,0)),(2,1))
            init_vec = dmrg.remish(init_vec)
        if dir == 'r':
            init_vec = np.tensordot(np.tensordot(self.mps.readS(site-1),self.mps.read(site),(0,1)),self.mps.read(site+1),(2,1))

        init_vec = np.reshape(init_vec,np.prod(init_vec.shape))
        
        if stage == None:
            En,grd = H.lanczos_grd(psi0=init_vec,exc=exc)
            grd_state = 1/np.sqrt(np.conj(grd)@grd)*grd
        if stage == 'Final':
            grd_state = 1/np.sqrt(init_vec@np.conj(init_vec))*init_vec
            En = np.conj(grd_state)@H.matvec(grd_state)
        
        grd_state = np.reshape(grd_state,(H.c1*H.d,H.d*H.c2))

        l,c,r = np.linalg.svd(grd_state,full_matrices=False)
        
        if len(c) > self.chi:
            l = l[:,:self.chi]
            c = c[:self.chi]
            r = r[:self.chi,:]

        ten_l = self.mps.left_ten(l)
        ten_r = self.mps.right_ten(r)

        self.mps.write(site,ten_l)
        self.mps.write(site+1,ten_r)
        self.mps.writeS(site,np.diag(c))

        if dir == 'r':
            self.cont.add(site,'l')
        
        if dir == 'l':
            self.cont.add(site+1,'r')
        
        return En, -c**2@np.log(c**2)
        
    def half_sweep(self,exc='off'):
        En = np.zeros(self.L//2-2)
        S = np.zeros(self.L//2-2)
        for site in range(self.L//2,self.L-2):
            En[site - self.L//2], S[site -  self.L//2] = self.step2sites(site,dir='r',exc=exc)
        
        return En, S
        
    def sweep(self,dir,exc='off'):
        En = np.zeros(self.L-4)
        S = np.zeros(self.L-4)
        
        for i in range(1,self.L-3):
            site = self.count[dir]*(i+1) + (1-self.count[dir])*(self.L - 3 -i)
            
            En[i-1], S[i-1] = self.step2sites(site,dir,exc=exc)
            print(site,site+1)
        return En, S 


    def remish(ten):
        d0,d1,d2,d3 = ten.shape
        res = np.zeros((d1,d0,d2,d3),dtype='complex')
        for i0 in range(d0):
            for i1 in range(d1):
                res[i1,i0,:,:] = ten[i0,i1,:,:]

        return res
    

    