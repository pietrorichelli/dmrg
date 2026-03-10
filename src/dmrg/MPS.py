import os 
import psutil
import shutil
import numpy as np

class MPS():
    """
    Matrix Product State (MPS) tensor storage class.
    Manages tensors for a 1D chain with automatic RAM/disk switching based on memory thresholds.
    
    Attributes:
        L : int 
            Number of sites in the chain
        d : int
            Physical dimension (default=2)
        S : dict
            Dictionary storing bond singular values (diagonal matrices) indexed by bond position
        path : str
            Directory path for storing disk-based tensors (default='MPS')
        ram : MPS.ram 
            RAM storage backend managing in-memory tensor dictionary
        disk : MPS.disk 
            Disk storage backend managing memmap-based tensors
    
    Methods:
        __init__(L, d, path, max_ram): 
            Initialize MPS with chain length L, physical dimension d, and memory threshold
        write(i, ten): 
            Write tensor to position i (RAM or disk based on memory threshold)
        read(i): 
            Read tensor from position i (automatically fetch from RAM or disk)
        writeS(i, ten): 
            Write bond singular values at position i to S dictionary
        readS(i): 
            Read bond singular values from position i
        write_left(i, mat): 
            Transform matrix to left-canonical form and write
        write_right(i, mat): 
            Transform matrix to right-canonical form and write
        delete(i): 
            Delete singular value files at bond i
        left_ten(mat): 
            Reshape matrix into left-canonical tensor form
        right_ten(mat): 
            Reshape matrix into right-canonical tensor form
        first_sweep(): 
            Iterator for initial half-sweep (right then left)
        sweep(): 
            Iterator for full sweep (right then left)
        right_sweep(): 
            Iterator for right-moving sweep only
        left_sweep(): 
            Iterator for left-moving sweep only
        random(): 
            Initialize MPS with random tensors and identity on center bond
    
    Inner Classes:
        ram: Manages in-memory dictionary storage with automatic spilling to disk
            - mps : dict 
                Dictionary of tensors indexed by site
            - max : int 
                Maximum RAM size in bytes before spilling to disk
            - write(i, ten): 
                Store tensor in memory
            - read(i): 
                Retrieve tensor from memory
            - empty_MPS(): 
                Initialize with boundary identities
        
        disk: Manages memmap-based file storage using .dat and .txt pairs
            - path : str 
                Base directory for storage
            - write(i, ten): 
                Write tensor to memmap file with shape metadata
            - read(i): 
                Load tensor from memmap file
            - shape(i): 
                Read tensor shape from .txt metadata file
            - empty_MPS(): 
                Initialize directory structure and boundary tensors
    """
    
    # Initialize MPS with chain length L, physical dimension d, and RAM/disk storage backends
    def __init__(self,L,d=2,path='MPS',max_ram=4):
        self.L = L 
        self.d = d
        self.S = {}
        self.path = path
        self.ram = MPS.ram(self)
        self.disk = MPS.disk(self)
        self.ram.max = max_ram*1024**3
        self.ram.current_size = 0

    # Write tensor at position i to RAM or disk based on memory threshold
    def write(self,i,ten):
        # Calculate tensor size in bytes
        tensor_size = ten.nbytes
        
        try:
            ((self.ram.current_size > self.ram.max - tensor_size) and self.disk.write or self.ram.write)(i,ten)
            self.ram.current_size += tensor_size * (self.ram.current_size <= self.ram.max - tensor_size)
        except ValueError:
            self.ram.current_size -= self.ram.mps.get(i, ten).nbytes if i in self.ram.mps else 0
            self.ram.mps.pop(i, None)
            self.disk.write(i,ten)
        except KeyError:
            self.disk.write(i,ten)

    # Read tensor from position i (RAM first, fallback to disk)
    def read(self,i):
        try: 
            return self.ram.read(i)
        except KeyError:
            return self.disk.read(i)

    # Write bond singular values (diagonal matrix) at position i
    def writeS(self,i,ten):
        self.S.update({i:ten})
    
    # Read bond singular values from position i
    def readS(self,i):
        return self.S[i]

    # Convert matrix to left-canonical tensor form and write to position i
    def write_left(self,i,mat):
        self.write(i,self.left_ten(mat))

    # Convert matrix to right-canonical tensor form and write to position i
    def write_right(self,i,mat):
        self.write(i,self.right_ten(mat))

    # Delete singular value files at bond i
    def delete(self,i):
        if os.path.isfile(self.path+f'/S/{i}-{i+1}.dat'):
            os.remove(self.path+f'/S/{i}-{i+1}.dat')
            os.remove(self.path+f'/S/{i}-{i+1}.txt')

    # Reshape matrix into left-canonical tensor form (physical index first)
    def left_ten(self,mat):
        # Reshape (a, b) → (a//d, d, b), then transpose to (d, a//d, b)
        return mat.reshape(mat.shape[0]//self.d, self.d, mat.shape[1]).transpose(1, 0, 2)

    # Reshape matrix into right-canonical tensor form (physical index first)
    def right_ten(self,mat):
        # Reshape (a, b) → (a, d, b//d), then transpose to (d, a, b//d)
        return mat.reshape(mat.shape[0], self.d, mat.shape[1]//self.d).transpose(1, 0, 2)

    # Iterator for initial half-sweep: right then left from center
    def first_sweep(self):
        half_right = [i for i in range(self.L//2 +self.L%2,self.L-2)]
        left = [self.L-4-i for i in range(self.L-4)]
        return zip(half_right+left,['r']*(self.L//2-2)+['l']*(self.L-4))

    # Iterator for full sweep: right then left
    def sweep(self):
        right = [i for i in range(2,self.L-2)]
        left = [self.L-4-i for i in range(self.L-4)]
        return zip(right+left,['r']*(self.L-4)+['l']*(self.L-4))

    # Iterator for right-moving sweep only
    def right_sweep(self):
        right = [i for i in range(2,self.L-2)]
        return zip(right,['r']*(self.L-4))

    # Iterator for left-moving sweep only
    def left_sweep(self):
        left = [self.L-4-i for i in range(self.L-4)]
        return zip(left,['l']*(self.L-4))

    # Initialize MPS with random tensors across chain and identity on center bond
    def random(self):
        ten = np.random.random((self.d,self.d,self.d))
        for i in range(1,self.L-1):
            self.write(i,ten)
        self.writeS(self.L//2-1+self.L%2,np.identity(self.d))

    class ram:
        # RAM storage backend for in-memory tensor dictionary with automatic disk spillover

        def __init__(self,parent):
            self.parent = parent
            self.max = self.max_size()

            # initialise the ram MPS
            self.mps = self.empty_MPS()
            # self.mps = {}
            

        @property
        def L(self):
            return self.parent.L

        @property
        def d(self):
            return self.parent.d

        
        # Calculate maximum tensor size that fits in available RAM
        def max_size(self):
            mem = psutil.virtual_memory()
            
            return int(np.sqrt(mem.available/(3*16*self.L*self.d)))

        # Store tensor in memory dictionary
        def write(self,i,ten):
            self.mps.update({i:ten})
        
        # Retrieve tensor from memory dictionary
        def read(self,i):
            return self.mps[i]
        
        # Initialize MPS dictionary with boundary identity tensors
        def empty_MPS(self):
            mps = {}
            mps.update({0:np.eye(self.d)})
            mps.update({self.L-1:np.eye(self.d)})
            return mps

    class disk:
        # Disk storage backend for tensors using memmap files with .dat and .txt metadata

        def __init__(self,parent):
            self.parent = parent
            self.empty_MPS()

        @property
        def L(self):
            return self.parent.L
        
        @property
        def d(self):
            return self.parent.d

        @property
        def path(self):
            return self.parent.path

        def empty_MPS(self):
            if os.path.isdir(self.path):
                shutil.rmtree(self.path)
            os.mkdir(self.path)
            for i in range(self.L):
                open(self.path+f'/ten_{i}.dat','w')
            self.write(0,np.eye(self.d))
            self.write(self.L-1,np.eye(self.d))
            if self.L%2 == 1:
                self.write(1,np.reshape(np.eye(self.d**2)[:,:self.d],(self.d,self.d,self.d)))

        # Write tensor to memmap file with shape metadata in .txt file
        def write(self,i,ten):
            f1 = np.memmap(self.path+f'/ten_{i}.dat',dtype='complex',mode='w+',shape=ten.shape)
            f1[:] = ten
            with open(self.path+f'/ten_{i}.txt','w') as f2: 
                f2.writelines(repr(ten.shape))
            del f1,f2

        
        # Read tensor shape metadata from .txt file
        def shape(self,i):
            s = open(self.path+f'/ten_{i}.txt','r')
            return eval(s.read())


        # Load tensor from memmap file using shape metadata
        def read(self,i):
            return np.memmap(self.path+f'/ten_{i}.dat',dtype='complex',mode='r',shape=self.shape(i))
