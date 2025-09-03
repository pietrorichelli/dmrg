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


    def mpo(self,p=None):

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

    def mpo(self,p=None):
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

    def mpo(self,p=None):

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

class MPO_AL():

    Id2 = np.identity(2)
    Id4 = np.identity(4)
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    Z = np.array([[1,0],[0,-1]])
    d = 4

    sub_BC = [0,2,3]
    sub_A = [1,4]

    def __init__(self,t_1,t_2,U,e_A,mu):
        self.t_1 = t_1
        self.t_2 = t_2
        self.U = U
        self.e_A = e_A
        self.mu = mu
        self.OP = [np.kron(MPO_AL.Sm,MPO_AL.Id2),np.kron(MPO_AL.Sp,MPO_AL.Id2),np.kron(MPO_AL.Z,MPO_AL.Sm),np.kron(MPO_AL.Z,MPO_AL.Sp)]
        self.coeff = [[self.t_1,self.t_2,self.t_2,0,0],
                    [self.t_1,self.t_2,self.t_1,0,self.t_2],
                    [0,self.t_2,self.t_1,0,self.t_2],
                    [0,self.t_1,self.t_2,0,self.t_2],
                    [0,self.t_1,0,0,self.t_2]]

    
    def mpo(self,p=None):

        MPO = np.zeros((4,4,22,22))

        MPO[:,:,0,0] = MPO_AL.Id4
        MPO[:,:,0,1] = np.kron(MPO_AL.Sp,MPO_AL.Z)
        MPO[:,:,0,6] = np.kron(MPO_AL.Sm,MPO_AL.Z)
        MPO[:,:,0,11] = np.kron(MPO_AL.Id2,MPO_AL.Sp)
        MPO[:,:,0,16] = np.kron(MPO_AL.Id2,MPO_AL.Sm)
        MPO[:,:,0,21] = -self.mu*(np.kron(MPO_AL.Z,MPO_AL.Id2) + np.kron(MPO_AL.Id2,MPO_AL.Z))

        # sublattice conditions
        if (p-1)%5 in MPO_AL.sub_BC:
            MPO[:,:,0,21] += self.U*np.kron(MPO_AL.Z,MPO_AL.Z)
        
        if (p-1)%5 in MPO_AL.sub_A:
            MPO[:,:,0,21] += self.e_A*(np.kron(MPO_AL.Z,MPO_AL.Id2) + np.kron(MPO_AL.Id2,MPO_AL.Z))

        for j in range(4):
            for i in range(4):
                MPO[:,:,1 +j*5 +i,2+j*5+i] = np.kron(MPO_AL.Z,MPO_AL.Z)

        
        for j in range(1,6):
            for i in range(4):
                MPO[:,:,j+5*i,21] = self.coeff[(p-1)%5][j-1]*self.OP[i]

        MPO[:,:,21,21] = MPO_AL.Id4

        return MPO

    def Wl(self):

        Wleft = np.zeros((4,4,22))

        Wleft[:,:,0] = MPO_AL.Id4
        Wleft[:,:,1] = np.kron(MPO_AL.Sp,MPO_AL.Z)
        Wleft[:,:,6] = np.kron(MPO_AL.Sm,MPO_AL.Z)
        Wleft[:,:,11] = np.kron(MPO_AL.Id2,MPO_AL.Sp)
        Wleft[:,:,16] = np.kron(MPO_AL.Id2,MPO_AL.Sm)
        Wleft[:,:,21] = (-self.mu + self.e_A )*(np.kron(MPO_AL.Z,MPO_AL.Id2) + np.kron(MPO_AL.Id2,MPO_AL.Z)) 

        return Wleft

    def Wr(self):

        Wright = np.zeros((4,4,22))

        for j in range(1,6):
            for i in range(4):
                Wright[:,:,j+5*i] = self.coeff[1][j-1]*self.OP[i]

        Wright[:,:,21] = MPO_AL.Id4

        return Wright
