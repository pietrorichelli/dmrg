import os 
import psutil
import shutil
import numpy as np

from .OptimizedTensorContractor import OptimizedTensorContractor

class CONT:
    """
    Contraction environment for DMRG.
    Stores left and right environment tensors with automatic RAM/disk switching.
    Used to build effective Hamiltonians for finite-system DMRG sweeps.
    
    Class Variables:
        dir : dict 
            Maps direction strings to subdirectory paths {'l':'/LEFT','r':'/RIGHT'}
        count : dict 
            Maps direction to index offset {'l':0,'r':1} for tensor contractions in add(i,ten,dir)
        add_dict : dict
            Stores Einstein summation equations for left/right environment contractions
    
    Attributes:
        mps : MPS 
            Reference to the Matrix Product State
        d : int 
            Physical dimension
        h : MPO
            The matrix product operator (Hamiltonian)
        L : int 
            Number of sites in the chain
        path : str 
            Directory path for storing contraction tensors (default='CONT')
        OTC : OptimizedTensorContractor
            Optimized tensor contraction engine for efficient environment updates
        ram : CONT.ram 
            RAM storage backend for left and right environments
        disk : CONT.disk 
            Disk storage backend for left and right environments
    
    Methods:
        __init__(mps, H, path, max_ram): 
            Initialize contraction environment with MPS and MPO
        write(i, ten, dir): 
            Write environment tensor at position i in direction dir (RAM or disk)
        read(i, dir): 
            Read environment tensor from position i in direction dir
        left(site): 
            Contract all tensors left of site (inclusive) to compute left environment
        right(site): 
            Contract all tensors right of site (inclusive) to compute right environment
        add(site, dir): 
            Update environment by adding one more site in the given direction
        env_prep(site): 
            Prepare left and right environments for two-site operation at site
        random(): 
            Initialize environments by adding sites incrementally from boundaries
    
    Inner Classes:
        ram: Manages in-memory dictionary storage for environments
            - LEFT (dict): 
                Dictionary of left environment tensors
            - RIGHT (dict): 
                Dictionary of right environment tensors
            - CONTS (dict): 
                Maps direction 'l'/'r' to LEFT/RIGHT dictionaries
            - write(i, ten, dir): 
                Store tensor in appropriate direction dictionary
            - read(i, dir): 
                Retrieve tensor from appropriate direction dictionary
            - empty_LEFT(): 
                Initialize LEFT with contraction from site 0
            - empty_RIGHT(): 
                Initialize RIGHT with contraction from site L-1
        
        disk: Manages memmap-based file storage for environments
            - path (str): 
                Base directory for LEFT and RIGHT subdirectories
            - write(site, ten, dir): 
                Write environment tensor to memmap file with shape metadata
            - read(site, dir): 
                Load environment tensor from memmap file
            - shape(site, dir): 
                Read tensor shape from .txt metadata file
            - empty_CONT(): 
                Initialize directory structure for LEFT and RIGHT contractions
    """

    dir = {'l':'/LEFT','r':'/RIGHT'}
    count = {'l':0,'r':1}
    add_dict = {'l':"abc,dae,dfbg,fch->egh",
                'r':"abc,dea,dfgb,fhc->egh"}

    # Initialize contraction environment with MPS and MPO, setting up RAM/disk storage backends
    def __init__(self,mps,H,path='CONT',max_ram=4):

        self.mps = mps
        self.d = mps.d
        self.h = H 
        self.L = mps.L
        self.path = path
        self.OTC = OptimizedTensorContractor()

        self.ram = CONT.ram(self)
        self.disk = CONT.disk(self)
        self.ram.max =  max_ram*1024**3 
        self.ram.current_size = 0

        pop_dict = {'l':self.ram.LEFT,'r':self.ram.RIGHT}

    # Write environment tensor at position i in direction dir (RAM if under threshold, else disk)
    def write(self,i,ten,dir):

        tensor_size = ten.nbytes
        
        try:
            ((self.ram.current_size > self.ram.max - tensor_size) and self.disk.write or self.ram.write)(i,ten,dir)
            self.ram.current_size += tensor_size * (self.ram.current_size <= self.ram.max - tensor_size)
        except ValueError:
            self.ram.current_size -= self.ram.CONTS[dir].get(i, ten).nbytes if i in self.ram.CONTS[dir] else 0
            self.ram.CONTS[dir].pop(i, None)
            self.disk.write(i,ten,dir)
        except KeyError:
            self.disk.write(i,ten,dir)


    # Read environment tensor from position i in direction dir (tries RAM first, falls back to disk)
    def read(self,i,dir):
        try:
            return self.ram.read(i,dir)
        except KeyError:
            return self.disk.read(i,dir)

    # Contract all tensors left of site (inclusive) to compute left environment from left boundary
    def left(self,site):
        h = self.h
        res = np.tensordot(np.tensordot(self.mps.read(0),h.Wl(),(0,0)),np.conj(self.mps.read(0)),(1,0))
        for i in range(1,site+1):
            res = self.OTC.contract("abc,dae,dfbg,fch->egh",*(res,self.mps.read(i),h.mpo(p=i),np.conj(self.mps.read(i))))

        return res

    # Contract all tensors right of site (inclusive) to compute right environment from right boundary
    def right(self,site):
        h = self.h
        res = np.tensordot(np.tensordot(self.mps.read(self.L-1),h.Wr(),(0,0)),np.conj(self.mps.read(self.L-1)),(1,0))
        for i in range(1,self.L-site):
            res = self.OTC.contract("abc,dea,dfgb,fhc->egh",*(res,self.mps.read(self.L-1-i),h.mpo(p=self.L-1-i),np.conj(self.mps.read(self.L-1-i))))

        return res

    
    # Add one more site to the environment in the given direction (left or right sweep)
    def add(self,site,dir):

        ten = self.read(site-(-1)**self.count[dir],dir)
        ten = self.OTC.contract(self.add_dict[dir],*(ten,self.mps.read(site),self.h.mpo(p=site),np.conj(self.mps.read(site))))

        self.write(site,ten,dir)
        return ten
    
    # Prepare left and right environments for two-site operation at site
    def env_prep(self,site):
        
        return self.read(site - 1,'l'),self.read(site + 2,'r')

    # Initialize environments by incrementally adding sites from both boundaries
    def random(self):
        for i in range(1,self.L//2):
            self.add(i+self.L%2,'l')
            self.add(self.mps.L-i-1,'r')
        
    class ram:
        # RAM storage backend for left and right environment tensors

        def __init__(self,parent):
            self.parent = parent

            self.LEFT = self.empty_LEFT()
            self.RIGHT = self.empty_RIGHT()

            self.CONTS = {'l': self.LEFT,'r':self.RIGHT}


        @property
        def L(self):
            return self.parent.L

        @property
        def d(self):
            return self.parent.d
        
        @property
        def h(self):
            return self.parent.h

        @property
        def mps(self):
            return self.parent.mps

        # Store environment tensor in RAM dictionary for given direction
        def write(self,i,ten,dir):
            self.CONTS[dir].update({i:ten})

        # Retrieve environment tensor from RAM dictionary for given position and direction
        def read(self,i,dir):
            return self.CONTS[dir][i]
    

        # Initialize left environment dictionary by contracting from left boundary
        def empty_LEFT(self):
            left = {}
            for i in range(self.L%2+1):
                left.update({i:self.parent.left(i)})
            return left

        # Initialize right environment dictionary by contracting from right boundary
        def empty_RIGHT(self):
            right = {}
            ten_r = np.tensordot(np.tensordot(self.mps.read(self.L-1),self.h.Wr(),(0,0)),np.conj(self.mps.read(self.L-1)),(1,0))
            right.update({self.L-1:ten_r})
            return right

    class disk:
        # Disk storage backend for large environment tensors using memmap files

        def __init__(self,parent):
            self.parent = parent
            self.path = self.parent.path
            self.empty_CONT()

        @property
        def L(self):
            return self.parent.L

        @property
        def d(self):
            return self.parent.d
        
        @property
        def h(self):
            return self.parent.h

        @property
        def mps(self):
            return self.parent.mps

        @property
        def dir(self):
            return self.parent.dir

        # Create and initialize directory structure for LEFT and RIGHT environment storage
        def empty_CONT(self):
            if os.path.isdir(self.path):
                shutil.rmtree(self.path)
            os.mkdir(self.path)
            os.mkdir(self.path+'/LEFT')
            os.mkdir(self.path+'/RIGHT')
            for i in range(self.L - 3):
                open(self.path+f'/LEFT/cont_{i}.dat','w')
                open(self.path+f'/RIGHT/cont_{self.L-i-1}.dat','w')

        # Write environment tensor to disk as memmap file with shape metadata
        def write(self,site,ten,dir):
            if dir != 'l' and dir != 'r':
                raise ValueError('the direction needs to be either l or r !!!')

            tenmap = np.memmap(self.path+self.dir[dir]+f'/cont_{site}.dat',dtype='complex',mode='w+',shape=ten.shape)
            tenmap[:] = ten
            with open(self.path+self.dir[dir]+f'/cont_{site}.txt','w') as f:
                f.writelines(repr(ten.shape))
            del tenmap,f

        # Read tensor shape from metadata file (.txt) for given site and direction
        def shape(self,site,dir):
            s = open(self.path+self.dir[dir]+f'/cont_{site}.txt','r')
            return eval(s.read())

        # Load environment tensor from disk memmap file using shape metadata
        def read(self,site,dir):
            return np.memmap(self.path+self.dir[dir]+f'/cont_{site}.dat',dtype='complex',mode='r',shape=self.shape(site,dir))