# dmrg

[![PyPI - Version](https://img.shields.io/pypi/v/dmrg.svg)](https://pypi.org/project/dmrg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dmrg.svg)](https://pypi.org/project/dmrg)

-----

A pure-Python implementation of the Density Matrix Renormalization Group (DMRG) algorithm for quantum systems on 1D chains. The library provides efficient tensor contraction, automatic memory management with RAM/disk switching, and a flexible framework for studying ground states and observables.

## Table of Contents

- [Installation](#installation)
- [Library Overview](#library-overview)
- [Examples](#Examples)
- [Documentation](#documentation)
- [License](#license)

## Installation

### Dependencies
Install required packages:
```console
python3 -m pip install numpy scipy psutil opt_einsum
```

### Install dmrg
```console
python3 -m pip install git+https://github.com/pietrorichelli/dmrg
```

Or install in development mode from source:
```console
cd /path/to/dmrg
python3 -m pip install -e .
```

## Library Overview

The `dmrg` library consists of six core modules:

### Core Classes

- **`MPS`** – Matrix Product State storage with automatic RAM/disk switching for large tensors
- **`CONT`** – Contraction environment managing left/right environments for DMRG sweeps
- **`MPO`** – Matrix Product Operator (Hamiltonian) interface with physics-specific subclasses
- **`EffH`** (in `lanczos`) – Effective Hamiltonian for two-site updates with Lanczos eigensolver
- **`dmrg`** – Main algorithm class orchestrating finite and infinite DMRG sweeps
- **`observables`** – Utilities for computing expectation values and correlations from optimized MPS

### Performance

- **`OptimizedTensorContractor`** – Caches optimal contraction paths using `opt_einsum` for repeated equations
- Automatic memory management with configurable RAM/disk spillover (default 4 GB threshold)
- All methods are fully documented with comprehensive docstrings

## Examples

The folder `examples/` contains ready-to-run code demonstrating the full API:

- **`TFIM.py`** – Command-line script that sweeps across the transverse field Ising model phase transition (21 field values)
- **`TFIM_guide.ipynb`** – Step-by-step interactive Jupyter notebook showing how to:
  - Initialize an MPS for a 1D chain
  - Construct an MPO for the transverse field Ising model
  - Run a complete DMRG sweep
  - Extract observables and entanglement entropy

### Running Examples

Command-line script (takes ~minutes depending on chain length):
```console
python3 examples/TFIM.py
```

Interactive notebook:
```console
jupyter lab examples/TFIM_guide.ipynb
```

## Documentation

All classes and methods include comprehensive docstrings accessible via Python help:
```python
from dmrg import MPS, CONT, dmrg
help(MPS)      # Matrix Product State
help(CONT)     # Contraction environment
help(dmrg)     # Main algorithm
```

For implementation details, see [copilot-instructions.md](copilot-instructions.md) which documents:
- High-level architecture and class relationships
- Tensor storage conventions and indexing
- Development workflows and benchmarking procedures
## License

`dmrg` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
