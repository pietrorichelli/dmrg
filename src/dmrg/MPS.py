import os
import shutil
import numpy as np

class MPS():
    """
        Class for an MPS
    """

    def __init__(self,L,mem='on',path='MPS',d=2,b_mps1=np.identity(2),b_mps2=None):
        self.L = L
        self.path = path
        self.d = d

        if mem == 'on':
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.mkdir(path)
            os.mkdir(path+'/S')
            for i in range(L):
                open(path+f'/ten_{i}.dat','w')
            self.write(0,b_mps1)
            if b_mps2 == None:
                b_mps2 = b_mps1
            self.write(L-1,b_mps2)

        if mem == 'off':
            raise ValueError("The option is not available turn the memory on")


    def __str__(self):
        return f'Matrix Product State of a ! D chain with {self.L} sites'
    
    def __repr__(self):
        return f'Matrix Product State of a ! D chain with {self.L} sites'


    def write(self,i,ten):
        f1 = np.memmap(self.path+f'/ten_{i}.dat',dtype='complex',mode='w+',shape=ten.shape)
        f1[:] = ten
        with open(self.path+f'/ten_{i}.txt','w') as f2: 
            f2.writelines(repr(ten.shape))
        del f1,f2

    def writeS(self,i,S):
        f1 = np.memmap(self.path+f'/S/{i}-{i+1}.dat',dtype='complex',mode='w+',shape=S.shape)
        f1[:] = S
        with open(self.path+f'/S/{i}-{i+1}.txt','w') as f2: 
            f2.writelines(repr(S.shape))
        del f1,f2
    
    def shape(self,i):
        s = open(self.path+f'/ten_{i}.txt','r')
        return eval(s.read())

    def shapeS(self,i):
        s = open(self.path+f'/S/{i}-{i+1}.txt','r')
        return eval(s.read())

    def read(self,i):
        return np.memmap(self.path+f'/ten_{i}.dat',dtype='complex',mode='r',shape=self.shape(i))
    
    def readS(self,i):
        return np.memmap(self.path+f'/S/{i}-{i+1}.dat',dtype='complex',mode='r',shape=self.shapeS(i))

    
    def write_bound(self,ten_l,ten_r=None):
        if ten_r == None:
            ten_r = ten_l
        self.write(0,ten_l)
        self.write(self.L-1,ten_r)


    def write_left(self,i,mat):
        self.write(i,self.left_ten(mat))

    def write_right(self,i,mat):
        self.write(i,self.right_ten(mat))

    
    def left_ten(self,mat):
        
        d = self.d
        a,b = mat.shape
        ten = np.zeros((d,int(a/d),b),dtype='complex')

        for i0 in range(d):
            for i1 in range(int(a/d)):
                ten[i0,i1,:] = mat[i1*d+i0,:]

        return ten

    
    def right_ten(self,mat):

        d = self.d
        a,b = mat.shape
        ten = np.zeros((d,a,int(b/d)),dtype='complex')

        for i0 in range(d):
            for i2 in range(int(b/2)):
                ten[i0,:,i2] = mat[:,i0*int(b/d)+i2]

        return ten
