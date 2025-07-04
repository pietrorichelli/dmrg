import numpy as np

class MPO_TFI():

    Id = np.identity(2)
    Z = np.array([[1,0],[0,-1]])
    X = np.array([[0,1],[1,0]])

    def __init__(self,J,h_x,pol=None,d=2):
        self.J = J 
        self.h_x = h_x 
        self.d = d
        self.pol = pol


    def Wl(self):

        Wleft = np.zeros((2,2,3))

        Wleft[:,:,0] = MPO_TFI.Id
        Wleft[:,:,1] = -self.J*MPO_TFI.Z
        Wleft[:,:,2] = -self.h_x*MPO_TFI.X

        if self.pol == 'tot':
            Wleft[:,:,2] -= 10*MPO_TFI.Z

        return Wleft


    def mpo(self):

        MPO = np.zeros((2,2,3,3))

        MPO[:,:,0,0] = MPO_TFI.Id
        
        MPO[:,:,0,1] = -self.J* MPO_TFI.Z
        MPO[:,:,0,2] = -self.h_x*MPO_TFI.X

        MPO[:,:,1,2] = MPO_TFI.Z
        MPO[:,:,2,2] = MPO_TFI.Id 

        return MPO


    def Wr(self):

        Wright = np.zeros((2,2,3))

        Wright[:,:,0] = -self.h_x*MPO_TFI.X
        Wright[:,:,1] = MPO_TFI.Z 
        Wright[:,:,2] = MPO_TFI.Id 

        if self.pol == 'tot':
            Wright[:,:,0] -= 10*MPO_TFI.Z

        return Wright
        

class MPO_ID():
    Id = np.identity(2)

    def __init__(self,d=2):
        self.d = d
    
    def Wl(self):
        Wleft = np.zeros((2,2,1))

        Wleft[:,:,0] = MPO_ID.Id

        return Wleft

    def mpo(self):
        MPO = np.zeros((2,2,1,1))

        MPO[:,:,0,0] = MPO_ID.Id

        return MPO

    def Wr(self):
        Wright = np.zeros((2,2,1))

        Wright[:,:,0] = MPO_ID.Id

        return Wright

class SUSY_MPO_1D():

    Id = np.identity(2)
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    Z = np.array([[1,0],[0,-1]])
    Pz = 1/2*(np.identity(2) - Z)
    d = 2
    
    def __init__(self,J=1):
        self.J = J


    def Wl(self):

        Wl = np.zeros((2,2,7))

        Wl[:,:,0] = self.Id
        Wl[:,:,1] = self.Pz
        Wl[:,:,2] = -self.J*self.Sp*self.Pz
        Wl[:,:,3] = -self.J*self.Pz*self.Sm
        Wl[:,:,6] = self.Pz

        return Wl

    def mpo(self):

        MPO = np.zeros((2,2,7,7))

        MPO[:,:,0,0] = self.Id
        MPO[:,:,0,1] = self.Pz

        MPO[:,:,1,2] = -self.J*self.Sp*self.Pz
        MPO[:,:,1,3] = -self.J*self.Pz*self.Sm

        MPO[:,:,2,4] = self.Pz*self.Sm 
        MPO[:,:,3,5] = self.Sp*self.Pz

        MPO[:,:,0,6] = 2*self.Pz
        MPO[:,:,4,6] = self.Pz 
        MPO[:,:,5,6] = self.Pz 
        MPO[:,:,6,6] = self.Id
        
        return MPO
    
    def Wr(self):
        
        Wr = np.zeros((2,2,7))

        Wr[:,:,2] = self.Pz*self.Sm 
        Wr[:,:,3] = self.Sp*self.Pz
        Wr[:,:,4] = self.Pz
        Wr[:,:,5] = self.Pz
        Wr[:,:,6] = self.Pz

        return Wr
