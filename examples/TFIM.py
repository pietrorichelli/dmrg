"""
    Python script that runs 21 points of the transverse field Ising model with polarized
    boundaries

"""

# import from dmrg
from dmrg.MPS import MPS 
from dmrg.MPO import MPO_TFI
from dmrg.CONT import CONT
from dmrg.dmrg import dmrg
from dmrg.obs import observables

# import standard packages
import numpy as np
import os 
import shutil

# Define folders where to store simulation data
path_mps = 'MPS'
path_cont = 'CONT'
path_out = 'OUT/'

# create folder out if not present
if os.path.isdir(path_out):
    shutil.rmtree(path_out)
os.mkdir(path_out)

# Define the parameters, system size and bond dimension
par = np.linspace(.1,2.1,21)
L = 21
chi = 100

for h_x in par:
    
    # define the par output
    path_par = path_out + f'out_{h_x:.2f}/' 
    os.mkdir(path_par)

    # initialise the MPS for the indicated chain length
    mps = MPS(L,max_ram=4) # max_ram sets how much memory can be used for the MPS in GB the standard is 4 GB

    # define the MPO 
    h = MPO_TFI(J=1,h_x=1,pol='tot')

    # define the contractions (it needs an mps and a MPO class as imputs)
    cont = CONT(mps=mps,H=h,max_ram=4) # max_ram sets how much memory can be used for the CONTRACTIONS in GB the standard is 4 GB

    # Initialize your dmrg (set low bond dimension to make the system grow faster)
    sys = dmrg(cont=cont,chi=10,cut=1e-12)

    # Grow the system up to the desired dimension
    En = sys.infinite()

    # open the energy sweep file and write the starting energy
    with open(path_par + 'E_sweep.txt','w') as f:
        f.write(f'{En[-1]} \n')

    # Increase the bond dimension to the desired one 
    sys.chi = chi

    # run the first one and half sweep
    for site,dir in mps.first_sweep():
        E,_,_ = sys.step2sites(site,dir=dir)
        
        # write sweep energy
        with open(path_par + 'E_sweep.txt','a') as f:
            f.write(f'{E} \n')

    # Set up counter and energy check
    k = 0 
    En_temp = np.zeros(2*L-8)
    En_temp[0] = En[-1]

    # Now we can sweep the system  (ideally until convergence)
    while np.abs(En_temp[0] - En_temp[-1]) > 1e-10:
        j = 0
        for site,dir in mps.sweep():
            En_temp[j],S,_ = sys.step2sites(site,dir=dir)

            # write sweep energy
            with open(path_par + 'E_sweep.txt','a') as f:
                f.write(f'{En_temp[j]} \n')

            j +=1

        # set maximum number of sweeps
        if k > 5:
            break
        
        k += 1

    # define observables
    obs = observables(mps)

    # Final sweep to store observables
    for site,dir in mps.right_sweep():
        _,S,_ = sys.step2sites(site,dir=dir,stage='Final')

        # Store local magnetization
        with open(path_par + 'Z.txt','a') as fz:
            if site == 2:
                o1,o2 = obs.bound_left(site-1,h.Z)
                fz.write(f'{site-2} {o1}\n')
                fz.write(f'{site-1} {o2}\n')

            fz.write(f'{site} {obs.single_site(site,h.Z).real} \n')

            if site == L-3:
                o1,o2 = obs.bound_right(site+1,h.Z)
                fz.write(f'{site+1} {o1}\n')
                fz.write(f'{site+2} {o2}')

        # Store entanglement entropy
        with open(path_par + 'S.txt','a') as fs:
            if site == 2:
                s0,s1 = obs.left_EE()
                fs.write(f'{0} {1} {s0}\n')
                fs.write(f'{1} {2} {s1}\n')

            fs.write(f'{site} {site+1} {S} \n')

            if site == L-3:
                fs.write(f'{site+1} {site+2} {obs.right_EE()}\n')

        # Store all two point correlations from site

        obs.all_corr(path_par + 'ZZ.txt',site,string=np.eye(2),obs1=h.Z)


    print(f'Parameter {h_x:.2f} done !!!')


