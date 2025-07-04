# dmrg

[![PyPI - Version](https://img.shields.io/pypi/v/dmrg.svg)](https://pypi.org/project/dmrg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dmrg.svg)](https://pypi.org/project/dmrg)

-----

## Table of Contents

- [Installation](#installation)
- [Examples](#Examples)
- [License](#license)

## Installation
Install dependencies 
```console
python3 -m pip install numpy
```
Install the dmrg software
```console
python3 -m  pip install git+https://github.com/pietrorichelli/dmrg
```

## Examples
The folder examples contains the `TFIM.py` script which will run 21 points across the transition for the transverse field Ising model. It can be run with:
```console
python3 TFIM.py
```

There is also a jupyter notebook `TFIM_guide.ipynb` that explains how to build a MPO class and a step by step guide on how to run a single point in parameter space of the transverse field Ising model
## License

`dmrg` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
