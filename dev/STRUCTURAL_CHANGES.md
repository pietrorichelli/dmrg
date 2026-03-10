# DMRG Package - Structural Changes Required

## Executive Summary

The `src/dmrg/` package has critical structural issues preventing proper functionality. This document outlines all necessary changes organized by file and priority.

---

## 🔴 CRITICAL ISSUES

### Issue 1: Missing `OptimizedTensorContractor.py` Module

**Status**: BLOCKING - Package cannot be imported

**Problem**:
- Both `dmrg.py` (line 7) and `CONT.py` import `OptimizedTensorContractor`
- The class exists only in `dev/dev.ipynb`, not as a standalone module
- This causes `ModuleNotFoundError` when trying to use the package

**Solution**: Create `src/dmrg/OptimizedTensorContractor.py`

**File Details**:
- **Location**: `src/dmrg/OptimizedTensorContractor.py`
- **Content**: Complete `OptimizedTensorContractor` class from `dev/dev.ipynb`
- **Key Features**:
  - Global `pop_idx` dictionary for efficient tensor popping
  - Path caching via `_cache` dictionary
  - `contract(equation, *tensors)` method with equation-based API
  - Internal `_precompute_plan()` method with `einsum_to_tensordot()` helper

**Code to Add**:
```python
import numpy as np
import opt_einsum as oe


class OptimizedTensorContractor:
    # [See dev/dev.ipynb for complete class code]
```

---

### Issue 2: Incomplete `__init__.py` Missing Public API

**Status**: CRITICAL - Package cannot be properly imported

**Problem**:
- Current `src/dmrg/__init__.py` only contains SPDX license header
- No exports of public classes: `dmrg`, `MPS`, `CONT`, `EffH`, `OptimizedTensorContractor`, etc.
- Users cannot do: `from dmrg import dmrg, MPS, CONT`
- No version information exposed

**Solution**: Update `src/dmrg/__init__.py` with proper exports

**File Content**:
```python
# SPDX-FileCopyrightText: 2025-present pietrorichelli <richelli.pietro@gmail.com>
#
# SPDX-License-Identifier: MIT

from .__about__ import __version__
from .dmrg import dmrg
from .MPS import MPS
from .CONT import CONT
from .lanczos import EffH
from .obs import observables
from .OptimizedTensorContractor import OptimizedTensorContractor
from .MPO import MPO_ID, MPO_TFI, SUSY_MPO_1D, MPO_AL

__all__ = [
    '__version__',
    'dmrg',
    'MPS',
    'CONT',
    'EffH',
    'observables',
    'OptimizedTensorContractor',
    'MPO_ID',
    'MPO_TFI',
    'SUSY_MPO_1D',
    'MPO_AL',
]
```

---

## 🟡 HIGH PRIORITY ISSUES

### Issue 3: `CONT.py` Missing Documentation Updates

**Status**: HIGH - Code works but documentation is incomplete

**Problem**:
- `CONT.py` class docstring missing:
  - `add_dict` class variable description
  - `OTC` (OptimizedTensorContractor) attribute description
- Methods lack descriptive comments
- Users cannot understand code purpose

**Solution**: Update `CONT.py` docstring and add method comments

**Changes Required**:

1. **Update Class Variables section** - Add:
```python
add_dict : dict
    Stores Einstein summation equations for left/right environment contractions
```

2. **Update Attributes section** - Add:
```python
OTC : OptimizedTensorContractor
    Optimized tensor contraction engine for efficient environment updates
```

3. **Add method comments** - Above each method definition:
```python
# Initialize contraction environment with memory management setup
def __init__(self,mps,H,path='CONT',max_ram=4):

# Write environment tensor to RAM or disk based on memory threshold
def write(self,i,ten,dir):

# Read environment tensor from RAM or disk storage
def read(self,i,dir):

# Compute left environment by contracting from site 0 to target site
def left(self,site):

# Compute right environment by contracting from site L-1 to target site
def right(self,site):

# Update environment by adding one more site in specified direction using optimized contraction
def add(self,site,dir):

# Retrieve prepared left and right environments for effective Hamiltonian calculation
def env_prep(self,site):

# Initialize environments by incrementally adding sites from chain boundaries
def random(self):
```

---

### Issue 4: `dmrg.py` Incomplete Documentation

**Status**: HIGH - Class docstring incomplete

**Problem**:
- Docstring only mentions `cont: Class DMRG.contractions` attribute
- Missing description for other important attributes:
  - `mps` (Matrix Product State)
  - `chi` (Bond dimension cutoff)
  - `h` (Hamiltonian/MPO)
  - `OTC` (Optimized tensor contractor)
  - etc.
- No method docstrings or descriptions

**Solution**: Expand class docstring and add method documentation

**Suggested Updates**:
```python
class dmrg():
    """
    Class that runs the DMRG algorithm on a 1 dimensional system.
    
    Attributes:
        cont : CONT
            Contraction environment containing MPS and related tensors
        mps : MPS
            Matrix Product State representation
        chi : int
            Bond dimension truncation cutoff
        h : MPO
            Matrix Product Operator (Hamiltonian)
        L : int
            Number of sites in the chain
        d : int
            Physical dimension
        k : int
            Krylov subspace dimension for Lanczos
        OTC : OptimizedTensorContractor
            Tensor contraction engine for efficient contractions
        
    Methods:
        infinite(): Run infinite DMRG algorithm for ground state
        step2sites(site, dir, stage): Execute two-site DMRG sweep step
    """
```

---

## 🟢 MEDIUM PRIORITY ISSUES

### Issue 5: Missing Error Handling in `CONT.py`

**Status**: MEDIUM - Code works but lacks robustness

**Problem**:
- Line 20: `pop_dict` is created but never used
- Variable scope issue: `pop_dict` defined in `__init__` but not stored as instance variable
- Line 21: `pop_dict[dir].pop(i)` in `write()` method will fail

**Solution**: Either remove unused `pop_dict` or fix implementation

**Option A** (Recommended - Remove):
```python
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
    # Remove: pop_dict = {'l':self.ram.LEFT,'r':self.ram.RIGHT}
```

---

### Issue 6: Inconsistent Naming in `OptimizedTensorContractor`

**Status**: MEDIUM - Code clarity issue

**Problem**:
- Class uses `pop_idx` as attribute name but variable names suggest `pop_strategy`
- In `dev/dev.ipynb`, original class used `pop_strategy` but `dev.ipynb` shows `pop_idx`
- Naming inconsistency makes code harder to understand

**Recommendation**: Standardize to `pop_strategy` throughout

---

## Implementation Priority

```
Priority 1 (BLOCKING):
  ✓ Create src/dmrg/OptimizedTensorContractor.py
  ✓ Update src/dmrg/__init__.py

Priority 2 (HIGH):
  ✓ Update CONT.py docstring and add method comments
  ✓ Update dmrg.py docstring and method documentation

Priority 3 (MEDIUM):
  ✓ Remove/fix pop_dict in CONT.py
  ✓ Rename pop_idx → pop_strategy for clarity

Priority 4 (NICE TO HAVE):
  ✓ Add docstrings to MPS.py inner classes
  ✓ Add docstrings to lanczos.py EffH class
  ✓ Add docstrings to obs.py observables class
```

---

## Testing Checklist

After implementing changes:

- [ ] `python -c "from dmrg import dmrg, MPS, CONT, OptimizedTensorContractor"` works
- [ ] `from dmrg import __version__` returns version string
- [ ] `import dmrg; print(dmrg.__all__)` shows all public exports
- [ ] CONT.py imports without errors
- [ ] dmrg.py imports without errors
- [ ] `pip install -e .` succeeds
- [ ] Run existing tests (if any)
- [ ] Run example scripts in `examples/`

---

## File-by-File Summary

| File | Status | Changes | Priority |
|------|--------|---------|----------|
| `src/dmrg/__init__.py` | INCOMPLETE | Add imports and exports | CRITICAL |
| `src/dmrg/OptimizedTensorContractor.py` | MISSING | Create new file | CRITICAL |
| `src/dmrg/CONT.py` | INCOMPLETE | Update docs, fix pop_dict | HIGH |
| `src/dmrg/dmrg.py` | INCOMPLETE | Expand docstring | HIGH |
| `src/dmrg/__about__.py` | OK | No changes | - |
| `src/dmrg/MPS.py` | OK | No changes required | - |
| `src/dmrg/lanczos.py` | OK | Add docstrings (optional) | MEDIUM |
| `src/dmrg/MPO.py` | OK | No changes required | - |
| `src/dmrg/obs.py` | OK | No changes required | - |

---

## Dependencies Verification

Ensure `pyproject.toml` includes:
```toml
[project]
dependencies = [
    "numpy",
    "scipy",
    "psutil",
    "opt_einsum",
]
```

---

## Notes

- The `OptimizedTensorContractor` class in `dev/dev.ipynb` is production-ready
- The class uses caching to avoid redundant path computations
- Global `pop_idx`/`pop_strategy` dictionary enables dictionary-based control flow
- All changes are backward compatible
- No breaking changes to public API

