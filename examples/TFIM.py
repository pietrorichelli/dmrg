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
from dmrg.logging import Logger

# import standard packages
import numpy as np
import matplotlib.pyplot as plt
import os 
import shutil
import time
import argparse
import glob


# Define folders where to store simulation data
path_mps = 'MPS'
path_cont = 'CONT'
path_out = 'OUT/'

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='DMRG study of transverse field Ising model',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  python TFIM_copy.py                           # Run with default parameters
  python TFIM_copy.py --h_min 0.5 --h_max 1.5 --num_points 11  # Custom range
  python TFIM_copy.py --L 30 --chi_max 150    # Longer chain with higher accuracy
  python TFIM_copy.py --max_sweeps 10         # More iterations
    '''
)

parser.add_argument('--h_min', type=float, default=0.1,
                    help='Minimum transverse field (default 0.1)')
parser.add_argument('--h_max', type=float, default=2.1,
                    help='Maximum transverse field (default 2.1)')
parser.add_argument('--num_points', type=int, default=21,
                    help='Number of parameter points (default 21)')
parser.add_argument('--L', type=int, default=21,
                    help='Chain length (default 21)')
parser.add_argument('--chi_max', type=int, default=100,
                    help='Maximum bond dimension (default 100)')
parser.add_argument('--k_max', type=int, default=300,
                    help='Maximum Lanczos k (default 300)')
parser.add_argument('--max_sweeps', type=int, default=5,
                    help='Maximum number of sweeps (default 5)')
parser.add_argument('--output_dir', type=str, default='OUT/',
                    help='Output directory (default OUT/)')

args = parser.parse_args()

# Use parsed arguments
path_out = args.output_dir

# create folder out if not present
if os.path.isdir(path_out):
    shutil.rmtree(path_out)
os.mkdir(path_out)

# Initialize logger
logger = Logger(path_out + 'run.log')

# Define the parameters, system size and bond dimension
par = np.linspace(args.h_min, args.h_max, args.num_points)
L = args.L
chi_max = args.chi_max
k_max = args.k_max
max_sweeps = args.max_sweeps

# Log simulation start
logger.print("="*70)
logger.print("TFIM DMRG Parameter Sweep")
logger.print("="*70)
logger.print(f"Parameters: {len(par)} points from {par[0]:.2f} to {par[-1]:.2f}")
logger.print(f"Chain length: L = {L}")
logger.print(f"Maximum bond dimension: chi = {chi_max}")
logger.print(f"Maximum Lanczos k: {k_max}")
logger.print(f"Maximum sweeps: {max_sweeps}")
logger.print("="*70)
logger.print("")

total_start = time.time()

for idx, h_x in enumerate(par, 1):
    
    param_start = time.time()
    logger.print(f"[{idx:2d}/{len(par)}] h_x = {h_x:.2f}  ", end='', flush=True)
    path_par = path_out + f'out_{h_x:.2f}/' 
    os.mkdir(path_par)

    # initialise the MPS for the indicated chain length
    mps = MPS(L,max_ram=4) # max_ram sets how much memory can be used for the MPS in GB the standard is 4 GB

    # define the MPO 
    h = MPO_TFI(J=1,h_x=1,pol='tot')

    # define the contractions (it needs an mps and a MPO class as imputs)
    cont = CONT(mps=mps,H=h,max_ram=4) # max_ram sets how much memory can be used for the CONTRACTIONS in GB the standard is 4 GB

    # Initialize your dmrg (set low bond dimension to make the system grow faster)
    sys = dmrg(cont=cont,chi=10,cut=1e-8)

    # Create random MPS and CONT
    mps.random()
    cont.random()

    # Increase the bond dimension to the desired one 
    # sys.chi = chi

    # run the first one and half sweep
    for site,dir in mps.first_sweep():
        _,_,_ = sys.step2sites(site,dir=dir,stage='Final')
        
        # # write sweep energy
        # with open(path_par + 'E_sweep.txt','a') as f:
        #     f.write(f'{E} \n')

    # Set up counter and energy check
    k = 0 
    En_temp = np.zeros(2*L-8)
    En_temp[0] = np.random.random()

    # Now we can sweep the system  (ideally until convergence)
    while np.abs(En_temp[0] - En_temp[-1]) > 1e-10 and k < max_sweeps: 
        j = 0
        
        # Increase the bond dimension
        sys.chi += chi_max//max_sweeps
        sys.chi = min([sys.chi,chi_max])

        # Increase the Krylov space dimension
        sys.k += k_max//max_sweeps
        sys.k = min([sys.k,k_max])

        for site,dir in mps.sweep():
            En_temp[j],S,_ = sys.step2sites(site,dir=dir)

            # write sweep energy
            with open(path_par + 'E_sweep.txt','a') as f:
                f.write(f'{En_temp[j]} \n')

            j += 1
        
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

    # Log completion of parameter point
    param_time = time.time() - param_start
    logger.print(f"✓ done  sweeps: {k}  ({param_time:.1f}s)")

# Cleanup and final summary
if os.path.isdir('MPS'):
    shutil.rmtree('MPS')
if os.path.isdir('CONT'):
    shutil.rmtree('CONT')

total_time = time.time() - total_start
logger.print("")
logger.print("="*70)
logger.print(f"Total runtime: {total_time/60:.1f} minutes")
logger.print(f"Output saved to: {path_out}")
logger.print("="*70)
logger.close()


# Find all E_sweep.txt files in ../examples/OUT/out_*
sweep_files = sorted(glob.glob(path_out+'/out_*/E_sweep.txt'))

fig, ax = plt.subplots(figsize=(12, 6))

for sweep_file in sweep_files:
    # Extract h_x value from path (e.g., '../examples/OUT/out_0.50/E_sweep.txt' -> '0.50')
    h_x = sweep_file.split('out_')[1].split('/')[0]
    
    # Load energy data
    data = np.loadtxt(sweep_file)

    ax.plot(np.log10(data-min(data)), label=f'h_x={h_x}', alpha=0.7)

ax.set_xlabel('DMRG Step', fontsize=12)
ax.set_ylabel('Energy (Log10)', fontsize=12)
ax.set_title('Energy Convergence Across Parameter Sweep', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.close()
fig.savefig(path_out+'/Convergence.png',bbox_inches='tight')
