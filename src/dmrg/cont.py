import os
import shutil
import numpy as np

class CONT():
    """
    Class that manages and stores in the memory the contractions on a 1 dimnasional MPS:
        Attributes:
            - mps: mps in the class DMRG.MPS
            - H: MPO of the hamiltonian of the class mpo_etc
            - mem: memory option if turned off the contraction are saved in a list (inconvinient and occupies a lot of memory)
            - path: folder path were the contractions are saved
    """

    # dictionary for the direction of the contraction
    dir = {'l':'/LEFT','r':'/RIGHT'}
    count = {'l':0,'r':1}


    def __init__(self,mps,H,mem='on',path='CONT'):
        self.mps = mps
        self.h = H 
        self.L = mps.L
        # build folder structure were tensors are saved
        if mem == 'on':
            self.path = path
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.mkdir(path)
            os.mkdir(path+'/LEFT')
            os.mkdir(path+'/RIGHT')
            for i in range(mps.L - 3):
                open(path+f'/LEFT/cont_{i}.dat','w')
                open(path+f'/RIGHT/cont_{mps.L-i-1}.dat','w')
            
            self.write_boundary()

        if mem == 'off':
            self.LEFT = []
            self.RIGHT = []
            
    # @classmethod

    def write(self,site,ten,dir):
        if dir != 'l' and dir != 'r':
            raise ValueError('the direction needs to be either l or r !!!')

        tenmap = np.memmap(self.path+self.dir[dir]+f'/cont_{site}.dat',dtype='complex256',mode='w+',shape=ten.shape)
        tenmap[:] = ten
        with open(self.path+self.dir[dir]+f'/cont_{site}.txt','w') as f:
            f.writelines(repr(ten.shape))
        del tenmap,f

    def shape(self,site,dir):
        s = open(self.path+self.dir[dir]+f'/cont_{site}.txt','r')
        return eval(s.read())

    def read(self,site,dir):
        return np.memmap(self.path+self.dir[dir]+f'/cont_{site}.dat',dtype='complex256',mode='r',shape=self.shape(site,dir))

    def add(self,site,dir):
        ten = self.read(site-(-1)**self.count[dir],dir)
        # ten = self.read(site,dir)

        ten = np.tensordot(ten,self.mps.read(site),(0,1+self.count[dir]))
        ten = np.tensordot(ten,self.h.mpo(p=site),([0,2],[2+self.count[dir],0]))
        ten = np.tensordot(ten,np.conj(self.mps.read(site)),([0,2],[1+self.count[dir],0]))

        self.write(site,ten,dir)
        return ten

    def left(self,site):
        """ 
        Contract all the tensors left of site including it 
        """
        h = self.h
        res = np.tensordot(np.tensordot(self.mps.read(0),h.Wl(),(0,0)),np.conj(self.mps.read(0)),(1,0))
        for i in range(1,site+1):
            res = np.tensordot(res,self.mps.read(i),(0,1))
            res = np.tensordot(res,h.mpo(p=i),([0,2],[2,0]))
            res = np.tensordot(res,np.conj(self.mps.read(i)),([0,2],[1,0]))

        return res
    
    def right(self,site):
        """
        Contract all the tensors right of site including it
        """
        h = self.h
        res = np.tensordot(np.tensordot(self.mps.read(self.L-1),h.Wr(),(0,0)),np.conj(self.mps.read(self.L-1)),(1,0))
        for i in range(1,self.L-site):
            res = np.tensordot(res,self.mps.read(self.L-1-i),(0,2))
            res = np.tensordot(res,h.mpo(p=self.L-1-i),([0,2],[3,0]))
            res = np.tensordot(res,np.conj(self.mps.read(self.L-1-i)),([0,2],[2,0]))
        
        return res

    def write_boundary(self):

        for i in range(self.L%2+1):
            ten_l = self.left(i)
            self.write(i,ten_l,'l')

        ten_r = np.tensordot(np.tensordot(self.mps.read(self.L-1),self.h.Wr(),(0,0)),np.conj(self.mps.read(self.L-1)),(1,0))
        self.write(self.L-1,ten_r,'r')
    
    
    
    def env_prep(self,site):
        
        return self.read(site - 1,'l'),self.read(site + 2,'r')

    def random(self):
        for i in range(1,self.L//2):
            self.add(i+self.L%2,'l')
            self.add(self.mps.L-i-1,'r')