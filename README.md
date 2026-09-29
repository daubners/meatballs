[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/daubners/meatballs/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/daubners/meatballs/actions/workflows/python-package.yml)

# meatballs

GPU-accelerated, microstructure-resolved simulations of anisotropic transport in polycrystalline NMC agglomerates.

<img src="meatballs.png" alt="Voxelized polycrystalline NMC agglomerate" width="520">

## Description

Commercial NMC cathode materials are commonly hierarchical agglomerates of many primary crystals. This improves manufacturability and tap density, but it is a substantial simplification to represent each agglomerate as a homogeneous sphere, as in the single-particle approximation of the Doyle–Fuller–Newman (DFN) model. The anisotropic transport of layered oxides means that crystal morphology and orientation can strongly influence the effective diffusivity. During battery ageing, cracking can further expose surfaces and shorten diffusion paths.

`meatballs` is a proof-of-concept framework for studying those effects directly on voxelized, polycrystalline agglomerates. It simulates anisotropic lithium transport on the GPU and supports virtual potentiostatic intermittent titration technique (PITT) experiments, providing a microstructure-resolved counterpart to the idealized spherical particle. The current focus is transport in fixed microstructures; cracking, mechanics, phase transformations, and coupling to a full cell model are research extensions rather than package features.

## Features

- Generate random or structured voxelized NMC agglomerates.
- Assign, estimate from particle shape, relabel, and export grain orientations.
- Define anisotropic crystal diffusivity and rotate each grain tensor into the reference frame.
- Construct a diffuse, arithmetic interpolation of the grain-resolved diffusivity tensor.
- Run GPU-accelerated PITT-style diffusion simulations with Evoxels grids and time steppers.
- Visualize scalar fields and orientation-aware slices interactively, and export fields to VTK.

## Installation

Requires Python 3.9 or newer. Clone the repository and install it in editable mode:

```bash
git clone git@github.com:daubners/meatballs.git
cd meatballs
python -m pip install -e .
```

For development tools, use `python -m pip install -e ".[dev]"`. Interactive Jupyter plotting additionally needs `ipywidgets` and `ipympl`.

## Usage

```python
import meatballs as mbs
from evoxels.timesteppers import PseudoSpectralIMEX

agglomerate = mbs.VoxelizedMeatball()
agglomerate.create_random_agglomerate(radius=10, num_seeds=20)

# Random grain angles are created with the agglomerate and may be replaced.
agglomerate.add_random_orientations()

solver = mbs.PITTSolver(
    agglomerate,
    diffusivity=[1.0, 1.0, 0.01],
    device="cuda",  # use "cpu" where CUDA is unavailable
    timestepper=PseudoSpectralIMEX,
)
solver.solve(
    time_increment=0.01,
    frames=10,
    max_iters=1000,
    bc=1,
    verbose="plot",
)

agglomerate.plot_slice(
    "concentration", agglomerate.Nz // 2, colormap="turbo"
)
```

## Reference

The physical model and virtual-PITT motivation are described in [Daubner et al., *Modeling Anisotropic Transport in Polycrystalline Battery Materials* (2023)](https://doi.org/10.3390/batteries9060310).

## License

This project is released under the [MIT License](LICENSE).
