# dmrg – Copilot Instructions

This document collects the small set of rules and architectural notes an
AI agent needs in order to be productive in `pietrorichelli/dmrg`.
The library is a mostly‑pure‑Python implementation of the density matrix
renormalization group (DMRG) algorithm on 1‑D chains.  Most of the heavy
math lives in `numpy`/`scipy` and the codebase is intentionally compact.

## High‑level architecture

* **`src/dmrg`** – the actual package.  There are only six Python files:
  * `MPS.py`, `CONT.py` – tensor storage and contraction environments
    respectively.  Both classes contain inner `ram` and `disk` helper classes
    that automatically switch between in‑memory dictionaries and `np.memmap`
    files when a size threshold is exceeded.  The `path` argument controls
the directory where `.dat`/`.txt` files are written.  Memory thresholds can be
changed by passing `max_ram` or by setting `psutil` limits; benchmarks in
`benchmark_mps_cont_memory.py` demonstrate the behaviour.
  * `MPO.py` – model‑specific matrix‑product operators.  There are a handful of
    hard‑coded subclasses (`MPO_TFI`, `MPO_ID`, `SUSY_MPO_1D`, `MPO_AL`) that
    expose `Wl()`, `mpo(p)`, and `Wr()` methods used by the DMRG routine.  Add
    new physics by adding a new subclass and following the existing template.
  * `lanczos.py` – the `EffH` class builds the effective Hamiltonian given left
    / right environments and exposes Lanczos iterations / gradient routines.
  * `dmrg.py` – wraps a `CONT` instance and runs the infinite/finite DMRG
    sweeps.  The public API is `infinite()` and `step2sites(site,dir,…)`; the
    latter returns energy and entanglement entropy.  `dmrg` imports `MPS_new`
    and `CONT_new` (see below) because benchmark scripts compare old/new
    versions.
  * `obs.py` – helper to compute single‑site expectation values and
    two‑point correlations from an `MPS` object.
  * `__about__.py` – stores version metadata used by `hatch`.

* **`examples/`** – ready‑to‑run code and a Jupyter guide (`TFIM_guide.ipynb`)
  showing how to initialize `MPS`, build an `MPO_TFI`, run a full sweep and
  extract observables.  `TFIM.py` is a command‑line script that loops over
  field values; it uses the same API as tests and is a good starting point for
  new experiments.

* **`benchmark_*.py`** – standalone runners used to compare the ``_old`` and
  ``_new`` implementations of `MPS`/`CONT` and to exercise memory‑threshold
  behaviour.  They are invoked as `python benchmark_mps.py 40 2 10` (L, d,
  runs) or `python benchmark_mps_cont_memory.py 20 2 1` and are the closest
  thing to a test suite in this repo.

* **`dev/`** – scratch notebooks and an alternate copy of the core classes
  (`dev/dmrg_src`).  They are not part of the published package but contain
  exploratory code; refer to them for examples of new features being
  prototyped.

## Conventions & patterns

* Sites are indexed `0..L-1`, with boundary identities stored at `0` and
  `L-1` in both `MPS` and `CONT`.
* Left/right directions are always `'l'`/`'r'` (see `CONT.dir` and the
  sweep iterators in `MPS`).
* Tensor files are created as `.dat` for the raw memmap and companion `.txt`
  holding the shape; any code that reads memmaps must call `shape()` first.
* The `S` dictionary in `MPS` holds the bond singular values (diagonal
  matrices).  Write with `writeS` and read with `readS`.
* `MPS.random()` and `CONT.random()` populate small tensors; they are handy for
  quick sanity checks in notebooks.
* When adding new models, update `examples/TFIM.py` or the notebooks to verify
  the full sweep works end‑to‑end.
* There is no automated test framework; use the benchmark scripts or create a
  small script/notebook to exercise new code.  Benchmarks generally clean up
  temporary directories by calling `.cleanup()`.

## Developer workflows

1. **Setup**
   ```sh
   cd /home/prichelli/Documents/Code/dmrg
   python3 -m pip install -e .       # or `pip install git+https://...`
   ```
   The project uses `hatch` (`pyproject.toml`) but no CI; building from source
   is normally unnecessary for local work.

2. **Running examples**
   ```sh
   python examples/TFIM.py             # runs 21 field values, outputs to OUT_*/
   jupyter lab examples/TFIM_guide.ipynb  # step‑by‑step interactive guide
   ```

3. **Benchmarks and memory tests**
   ```sh
   python benchmark_mps.py <L> <d> <runs>
   python benchmark_mps_cont_memory.py <L> <d> [factor]
   ```
   Modify or extend these scripts when optimizing or refactoring `MPS`/`CONT`.
   They also document the public API for the two classes.

4. **Debugging / exploration**
   * Open `dev/dev.ipynb` to run arbitrary code against `dev/dmrg_src` classes
     (which mirror the ones in `src/` with `_new` suffixes).
   * Use `psutil` to inspect memory; `cont.ram.max` gives the point at which
     writes start spilling to disk.

5. **Packaging & publishing**
   * Version is controlled in `src/dmrg/__about__.py`.  `hatchling` is used to
     build wheels for PyPI; you rarely need to touch this unless bumping
     versions.

## Notes for AI agents

* Focus on the finite‑DMRG sweep logic in `dmrg.step2sites` when changing
  physics; the environment update (`cont.add`) is central.
* When editing storage classes (`MPS`/`CONT`), make sure the RAM/disk
  symmetry is preserved and that memory thresholds (`max_ram`) behave as in
  the benchmark scripts.  The two implementations labelled `*_new` in the
  repository are experimental clones and may not exist in `src/` after merging
  – adapt import paths accordingly.
* Keep external dependencies minimal: only `numpy`, `scipy` and `psutil` are
  required; avoid adding heavy libraries.
* There is no test harness; new functionality should be exercised via the
  `examples` directory or by adding to the benchmark scripts.
* Use the `MPS.*_sweep()` iterators to drive sweeps; they encode boundary
  conditions and should not be reimplemented ad‑hoc.

> **TL;DR:** treat `src/dmrg` as a small physics library composed of
MPS/CONT storage, MPO models, an `EffH` Lanczos engine and the finite/infinite
DMRG driver.  Most development consists of adding new MPOs or extending the
algorithms in `dmrg.py`; everything else is utility code for IO and
observables.  Benchmarks double as informal tests and are the best place to
look for usage examples.

Feel free to ask the author for clarification on missing `*_new` files or
expected performance targets.