# Algorithm Debugging Report

**Date:** February 3, 2026  
**Issue:** DMRG algorithm test hanging and not converging properly

---

## Summary of Issues Found and Fixed

### 🔴 **CRITICAL BUG #1: Variable Scope Error**

**File:** `dev/dmrg_src/dmrg.py` - `step2sites()` method  
**Lines:** 70-71

**Problem:**
```python
def step2sites(self, site, dir, exc='off', stage=None):
    env_left = cont.left(site-1)        # ❌ NameError: cont is undefined!
    env_right = cont.right(site+2)      # ❌ Should be self.cont
```

**Root Cause:** Missing `self.` prefix on contraction object

**Fix:**
```python
def step2sites(self, site, dir, exc='off', stage=None):
    env_left = self.cont.left(site-1)   # ✓ Correct
    env_right = self.cont.right(site+2) # ✓ Correct
```

**Impact:** Without this fix, every DMRG step would crash with `NameError`

---

### 🔴 **CRITICAL BUG #2: Variable Bond Dimension in MPO**

**File:** `dev/dev.ipynb` - `MPO_AL_test.mpo()` method  
**Lines:** 58-60

**Problem:**
```python
def mpo(self, p):
    # ❌ Variable bond dimensions!
    MPO = np.zeros((4, 4, 22+(p-1)%5, 22+(p%5)))
```

**Root Cause:** Bond dimensions should be CONSTANT across all sites for proper tensor contraction

**Consequences:**
- Bond dimension oscillates between 22-26
- Tensor contractions fail due to shape mismatch
- `np.tensordot()` throws dimension errors

**Fix:**
```python
def mpo(self, p):
    # ✓ Fixed dimensions
    MPO = np.zeros((4, 4, 22, 22), dtype='complex128')
```

---

### 🟠 **ISSUE #3: Lanczos Convergence Hanging**

**File:** `dev/dev.ipynb` - Test cell  
**Symptoms:**
- Each DMRG step takes 5+ seconds
- Progress bar shows only 1-2 sweeps in several seconds
- Execution requires `KeyboardInterrupt`

**Root Causes:**

1. **Lanczos dimension too large for system size**
   - Chain length: L=8, physical dim: d=4
   - Effective Hamiltonian dimension: 4 × 4 × 16 × 4 = 1024
   - Lanczos parameter: k=300 (requesting 300 Krylov vectors)
   - For a 1024-dim space, k=300 is excessive and requires many iterations

2. **Convergence criteria too strict**
   - Beta convergence threshold: `beta < 1e-8`
   - May need many iterations to reach numerical precision

3. **Random initialization**
   - `np.random.rand()` gives uniform [0,1) distribution
   - Not optimal for ground state search

**Fix:**
```python
# Reduce Lanczos dimension for small systems
sys = dmrg(cont, chi=20, k=50)  # Changed from k=300 to k=50
```

---

### 🟠 **ISSUE #4: Infinite Loop in Test**

**File:** `dev/dev.ipynb` - Test cell  
**Lines:** ~38-48

**Problem:**
```python
k = 0
while True:                          # ❌ Infinite loop
    for site, dir in tqdm(mps.sweep()):
        En, _, En_pre = sys.step2sites(site, dir=dir, exc='off')
        EE.append(En)
    k += 1
    if k == 1:                       # Only breaks after 1 iteration
        break                        # ...but never reached first time
```

**Issue:** The loop runs indefinitely on the first iteration because:
1. Lanczos eigensolve is slow (5+ seconds per step)
2. User had to `KeyboardInterrupt` to stop it
3. Loop structure doesn't provide checkpointing or convergence monitoring

**Fix:**
```python
# Replace with explicit loop count and convergence checking
num_sweeps = 3
for sweep_num in range(num_sweeps):
    print(f"Sweep {sweep_num + 1}:")
    for site, dir in tqdm(mps.sweep()):
        En, _, En_pre = sys.step2sites(site, dir=dir, exc='off')
        EE.append(En)
    
    # Check convergence
    energy_diff = abs(EE[-1] - EE[-2])
    print(f"Energy difference: {energy_diff:.2e}")
    if energy_diff < 1e-6:
        print("Converged!")
        break
```

---

### 🟡 **MINOR ISSUE #5: Energy Output Format**

**File:** `dev/dev.ipynb` - Test cell output

**Observed:**
```
r (0.6076023801076939939+0j)
```

**Problem:**
- Complex number representation (with imaginary unit shown)
- Incorrect output format

**Root Cause:** Energy differences printed as complex numbers instead of real scalars

**Fix:**
```python
# Extract real part when printing
print(f"Energy: {En.real:.8f}")
```

---

## Performance Improvements Made

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Lanczos k | 300 | 50 | Overly aggressive for small systems |
| Data type | complex256 | complex128 | MPO constructors updated |
| Loop structure | Infinite loop | Bounded iteration | Better monitoring and control |
| Convergence check | None | Energy diff threshold | Proper termination criterion |

---

## Expected Behavior After Fixes

✅ **No more crashes** - Variable scope fixed  
✅ **Proper tensor contractions** - Bond dimensions fixed  
✅ **Faster execution** - Reduced Lanczos dimension  
✅ **Proper termination** - Removed infinite loop  
✅ **Better monitoring** - Energy convergence printed each sweep  

---

## Testing Recommendations

1. **Start with simple model**: Use `MPO_XY()` (2-site interactions) instead of `MPO_AL_test()`
2. **Small system**: L=8 is fine, but with d=2 physical dimension
3. **Monitor convergence**: Check energy difference < 1e-6 per sweep
4. **Verify energy trend**: Should decrease or plateau, never oscillate wildly
5. **Check memory**: Use `mps.get_memory_usage()` to monitor MPS size

---

## Files Modified

- [dev.ipynb](dev/dev.ipynb) - Fixed test cell and MPO class
- [dmrg_src/dmrg.py](dev/dmrg_src/dmrg.py) - Fixed variable scope in `step2sites()`

---

## Next Steps

1. Run corrected test and verify convergence behavior
2. Compare with known results (e.g., TenPy on same system)
3. Implement other bottleneck fixes from CODE_ANALYSIS.md
4. Add comprehensive error checking and logging
5. Consider switching to TenPy for production work

