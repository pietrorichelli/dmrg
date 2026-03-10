"""
Transverse Field Ising Model (TFIM) - Parameter Sweep with DMRG

This script performs a complete DMRG study of the 1D transverse field Ising model
across the quantum phase transition:

    H = -J Σ σ_i^z σ_{i+1}^z - h_x Σ σ_i^x

PHASE TRANSITION:
    - Ferromagnetic phase (h_x < 1): Ground state has broken Z2 symmetry
    - Paramagnetic phase (h_x > 1): Ground state is unique, field-aligned
    - Critical point at h_x = 1: Quantum critical point (conformal field theory)

PARAMETER RANGE:
    h_x ∈ [0.1, 2.1] (21 points) captures both phases and critical region
    
OUTPUT FILES (per parameter point):
    - E_sweep.txt: Energy at each DMRG step (convergence monitor)
    - Z.txt: Local magnetization <σ^z_i> at each site
    - S.txt: Entanglement entropy S(i,i+1) between bonds
    - ZZ.txt: Two-point correlations <σ^z_i σ^z_j> (correlation length)

DMRG STRATEGY:
    1. Initialize with low bond dimension (chi=10) for fast system growth
    2. Increase to high dimension (chi=100) for accurate ground state
    3. Sweep until energy converges (tolerance 1e-10, max 5 sweeps)
    4. Extract observables in final sweep

"""

import argparse
import sys
import time
import numpy as np
import os 
import shutil

# import from dmrg
from dmrg.MPS import MPS 
from dmrg.MPO import MPO_TFI
from dmrg.CONT import CONT
from dmrg.dmrg import dmrg
from dmrg.obs import observables
from dmrg.logging import Logger


def create_parameter_grid(h_min=0.1, h_max=2.1, num_points=21):
    """Create logarithmic or linear grid of transverse field parameters.
    
    Parameters
    ----------
    h_min, h_max : float
        Range of h_x values to sample
    num_points : int
        Number of parameter points
        
    Returns
    -------
    np.ndarray
        Array of h_x values spanning the phase transition region
    """
    return np.linspace(h_min, h_max, num_points)


def run_dmrg_simulation(h_x, L=21, chi_init=10, chi_final=100, 
                        cut=1e-8, conv_tol=1e-10, max_sweeps=5, max_ram=4):
    """Run DMRG calculation for single parameter point.
    
    Parameters
    ----------
    h_x : float
        Transverse field strength
    L : int
        Chain length (default 21)
    chi_init : int
        Initial bond dimension for MPS growth (default 10)
    chi_final : int
        Final bond dimension for optimization (default 100)
    cut : float
        Singular value cutoff threshold (default 1e-8)
    conv_tol : float
        Energy convergence tolerance (default 1e-10)
    max_sweeps : int
        Maximum DMRG sweeps (default 5)
    max_ram : float
        RAM threshold in GB before disk spillover (default 4 GB)
        
    Returns
    -------
    tuple
        (mps, h, cont, sys, obs) objects for observable extraction
    """
    # Initialize MPS for chain of length L
    # Tensors stored in 'MPS/' directory with automatic RAM->disk spillover
    mps = MPS(L, max_ram=max_ram)
    
    # Define the MPO for TFIM with polarized boundaries
    # J=1 (nearest-neighbor coupling), h_x (transverse field), pol='tot' (boundary condition)
    h = MPO_TFI(J=1, h_x=h_x, pol='tot')
    
    # Initialize contraction environment for efficient two-site updates
    # CONT manages left/right environments with OptimizedTensorContractor (caches paths)
    cont = CONT(mps=mps, H=h, max_ram=max_ram)
    
    # Initialize DMRG with low bond dimension for rapid system growth
    sys = dmrg(cont=cont, chi=chi_init, cut=cut)
    
    # ========== PHASE 1: System Growth (infinite DMRG) ==========
    # Grow MPS from center: L/2 -> L sites with low bond dimension
    En = sys.infinite()
    
    # ========== PHASE 2: Increase Accuracy ==========
    # Set high bond dimension for ground state optimization
    sys.chi = chi_final
    
    # First half-sweep to canonical form (after growth from center)
    # site,dir yields (site_index, direction='r' or 'l')
    k_increased_events = []
    for site, dir in mps.first_sweep():
        E, _, _ = sys.step2sites(site, dir=dir)
        if sys.k_increased:
            k_increased_events.append(f"k↑@sweep0:site{site}")
    
    # ========== PHASE 3: Variational Optimization ==========
    # Sweep back-and-forth until energy converges
    k = 0
    En_temp = np.zeros(2*L-8)
    En_temp[0] = En[-1]
    
    # Convergence: energy change per sweep < 1e-10, capped at 5 sweeps
    while np.abs(En_temp[0] - En_temp[-1]) > conv_tol and k < max_sweeps:
        j = 0
        # Full sweep: right->left then left->right
        for site, dir in mps.sweep():
            En_temp[j], S, _ = sys.step2sites(site, dir=dir)
            if sys.k_increased:
                k_increased_events.append(f"k↑@sweep{k+1}:site{site}")
            j += 1
        k += 1
    
    # Create observables object for extraction
    obs = observables(mps)
    
    return {
        'energy': En_temp[-1].real,
        'sweeps_needed': k,
        'converged': np.abs(En_temp[0] - En_temp[-1]) < conv_tol,
        'mps': mps,
        'h': h,
        'cont': cont,
        'sys': sys,
        'obs': obs,
        'k_increased_events': k_increased_events
    }


# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DMRG study of transverse field Ising model phase transition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
                Examples:
                python TFIM.py                           # Run full 21-point sweep (h_x: 0.1-2.1)
                python TFIM.py --h_min 0.5 --h_max 1.5 --num_points 11  # Custom range
                python TFIM.py --L 30 --chi_final 150   # Longer chain with higher accuracy
                python TFIM.py --max_ram 2              # Limit memory to 2 GB
                '''
    )
    
    # Major parameters with physical interpretation
    parser.add_argument('--h_min', type=float, default=0.1,
                        help='Minimum transverse field (low→ferromagnetic phase)')
    parser.add_argument('--h_max', type=float, default=2.1,
                        help='Maximum transverse field (high→paramagnetic phase)')
    parser.add_argument('--num_points', type=int, default=21,
                        help='Number of parameter points (default 21 spans transition well)')
    parser.add_argument('--L', type=int, default=21,
                        help='Chain length (default 21; larger=more accurate but slower)')
    
    # DMRG algorithm parameters
    parser.add_argument('--chi_init', type=int, default=10,
                        help='Initial bond dimension for growth (low=fast, default 10)')
    parser.add_argument('--chi_final', type=int, default=100,
                        help='Final bond dimension for optimization (high=accurate, default 100)')
    parser.add_argument('--cut', type=float, default=1e-8,
                        help='Singular value cutoff (default 1e-8, smaller=more accurate)')
    
    # Convergence parameters
    parser.add_argument('--conv_tol', type=float, default=1e-10,
                        help='Energy convergence tolerance (default 1e-10)')
    parser.add_argument('--max_sweeps', type=int, default=5,
                        help='Maximum DMRG sweeps (default 5, prevents excessive runs)')
    
    # Resource parameters
    parser.add_argument('--max_ram', type=float, default=4,
                        help='RAM threshold in GB before disk spillover (default 4)')
    parser.add_argument('--output_dir', type=str, default='OUT/',
                        help='Output directory for results (default OUT/)')
    
    args = parser.parse_args()
    
    # Create output directory
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    
    # ========== SETUP LOGGING ==========
    # Initialize logger to write output to both terminal and file
    logger = Logger(args.output_dir + 'run.log')
    
    # ========== SETUP ==========
    logger.print("="*70)
    logger.print("TFIM DMRG Phase Transition Study")
    logger.print("="*70)
    logger.print(f"\nParameters:")
    logger.print(f"  h_x range: [{args.h_min:.2f}, {args.h_max:.2f}] ({args.num_points} points)")
    logger.print(f"  Chain length: L = {args.L}")
    logger.print(f"  Bond dimensions: {args.chi_init} → {args.chi_final}")
    logger.print(f"  Convergence tolerance: {args.conv_tol:.0e}")
    logger.print(f"  Max RAM: {args.max_ram} GB")
    logger.print("="*70)
    logger.print("")
    
    # Create parameter grid spanning phase transition
    # Critical point at h_x ≈ 1.0
    par = create_parameter_grid(args.h_min, args.h_max, args.num_points)
    
    logger.print(f"Output directory: {args.output_dir}")
    logger.print(f"Log file: {args.output_dir}run.log")
    logger.print("="*70)
    logger.print("")
    
    # ========== PARAMETER SWEEP ==========
    total_start = time.time()
    results = []
    
    for idx, h_x in enumerate(par, 1):
        param_start = time.time()
        
        try:
            logger.print(f"[{idx:2d}/{args.num_points}] h_x = {h_x:.2f}  ", end='', flush=True)
            
            # Create output directory for this parameter
            path_par = args.output_dir + f'out_{h_x:.2f}/'
            os.mkdir(path_par)
            
            # ========== RUN DMRG ==========
            sim_result = run_dmrg_simulation(
                h_x=h_x,
                L=args.L,
                chi_init=args.chi_init,
                chi_final=args.chi_final,
                cut=args.cut,
                conv_tol=args.conv_tol,
                max_sweeps=args.max_sweeps,
                max_ram=args.max_ram
            )
            
            # Extract objects from result
            mps = sim_result['mps']
            h = sim_result['h']
            cont = sim_result['cont']
            sys = sim_result['sys']
            obs = sim_result['obs']
            k_events = sim_result['k_increased_events']
            
            # ========== EXTRACT OBSERVABLES ==========
            # Final sweep to collect physical quantities
            # Opens files once and writes all data (more efficient than repeated open/close)
            with open(path_par + 'Z.txt', 'w') as fz, \
                 open(path_par + 'S.txt', 'w') as fs:
                
                for site, dir in mps.right_sweep():
                    _, S, _= sys.step2sites(site, dir=dir, stage='Final')
                    
                    # ========== Local Magnetization ==========
                    # <σ^z_i> at each site (order parameter for ferromagnetic phase)
                    if site == 2:
                        # Boundary corrections: magnetization at sites 0 and 1
                        o1, o2 = obs.bound_left(site-1, h.Z)
                        fz.write(f'{site-2} {o1.real:.12e}\n')
                        fz.write(f'{site-1} {o2.real:.12e}\n')
                    
                    fz.write(f'{site} {obs.single_site(site, h.Z).real:.12e}\n')
                    
                    if site == args.L-3:
                        # Boundary corrections at right edge
                        o1, o2 = obs.bound_right(site+1, h.Z)
                        fz.write(f'{site+1} {o1.real:.12e}\n')
                        fz.write(f'{site+2} {o2.real:.12e}\n')
                    
                    # ========== Entanglement Entropy ==========
                    # S(i) = -Tr(ρ log ρ) on bond i,i+1 (detects criticality)
                    if site == 2:
                        s0, s1 = obs.left_EE()
                        fs.write(f'{0} {1} {s0.real:.12e}\n')
                        fs.write(f'{1} {2} {s1.real:.12e}\n')
                    
                    fs.write(f'{site} {site+1} {S.real:.12e}\n')
                    
                    if site == args.L-3:
                        fs.write(f'{site+1} {site+2} {obs.right_EE().real:.12e}\n')
                    
                    # ========== Two-Point Correlations ==========
                    # <σ^z_i σ^z_j> for all j>i (measures correlation length)
                    obs.all_corr(path_par + 'ZZ.txt', site, string=np.eye(2), obs1=h.Z)
            
            # ========== TIMING & STATUS ==========
            param_time = time.time() - param_start
            status = "✓ CONV" if sim_result['converged'] else "⨯ NCONV"
            k_info = f"  k_increases: {len(k_events)}" if k_events else ""
            logger.print(f"E = {sim_result['energy']:10.8f}  sweeps: {sim_result['sweeps_needed']}  {status}  ({param_time:.1f}s){k_info}")
            
            if k_events:
                logger.print(f"  Events: {', '.join(k_events)}")
            
            results.append({
                'h_x': h_x,
                'energy': sim_result['energy'],
                'converged': sim_result['converged'],
                'sweeps': sim_result['sweeps_needed'],
                'time': param_time,
                'k_increases': len(k_events)
            })
        
        except Exception as e:
            logger.print(f"✗ FAILED: {str(e)}")
            results.append({
                'h_x': h_x,
                'energy': None,
                'converged': False,
                'sweeps': None,
                'time': time.time() - param_start,
                'error': str(e)
            })
            continue
    
    # ========== CLEANUP ==========
    # Remove temporary tensor storage to free disk space after extraction
    if os.path.isdir('MPS'):
        shutil.rmtree('MPS')
    if os.path.isdir('CONT'):
        shutil.rmtree('CONT')
    
    # ========== SUMMARY ==========
    total_time = time.time() - total_start
    successful = sum(1 for r in results if r['converged'])
    
    logger.print("\n" + "="*70)
    logger.print("SUMMARY")
    logger.print("="*70)
    logger.print(f"Total runtime: {total_time/60:.1f} minutes")
    logger.print(f"Successful: {successful}/{len(results)} parameter points converged")
    logger.print(f"Output saved to: {args.output_dir}")
    logger.print("="*70 + "\n")
    
    # Save summary of all parameter points
    with open(args.output_dir + 'summary.txt', 'w') as f:
        f.write(f"h_x\tEnergy\tConverged\tSweeps\tTime(s)\tk_increases\n")
        for r in results:
            status = "Yes" if r['converged'] else "No"
            energy_str = f"{r['energy']:.12e}" if r['energy'] is not None else "FAILED"
            sweeps_str = str(r['sweeps']) if r['sweeps'] is not None else "N/A"
            k_inc = r.get('k_increases', 0)
            f.write(f"{r['h_x']:.2f}\t{energy_str}\t{status}\t{sweeps_str}\t{r['time']:.2f}\t{k_inc}\n")
    
    # Close log file
    logger.close()


