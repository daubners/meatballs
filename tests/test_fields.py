"""Tests for field handling with VoxelFields class."""

import numpy as np
import meatballsss as mbs
from evoxels import VoxelFields

def test_voxelFields_init():
    test = np.ones((2,3,4))
    agglomerate = mbs.VoxelizedMeatball(grain_map=test)
    assert isinstance(agglomerate, VoxelFields)
    assert (agglomerate.Nx, agglomerate.Ny, agglomerate.Nz) == (2,3,4)
    assert agglomerate.grid_info().shape == (2,3,4)
    assert mbs.VoxelizedMeatball.plot_slice is VoxelFields.plot_slice
    assert mbs.VoxelizedMeatball.plot_field_interactive is VoxelFields.plot_field_interactive

def test_voxelFields_init_domain():
    test = np.ones((2,3,4))
    agglomerate = mbs.VoxelizedMeatball(grain_map=test)
    assert (agglomerate.domain_size, agglomerate.spacing) == ((2,3,4),(1, 1, 1))

def test_voxelFields_init_spacing():
    test = np.ones((2,3,4))
    agglomerate = mbs.VoxelizedMeatball(grain_map=test, spacing=(1,3,7))
    assert (agglomerate.domain_size, agglomerate.spacing) == ((2,9,28), (1,3,7))

def test_voxelFields_init_fields():
    a = mbs.VoxelizedMeatball()
    a.create_random_agglomerate(3, 3)
    a.add_field("c", 0.123*np.ones((a.Nx, a.Ny, a.Nz)))

    assert (a.fields['c'][1,2,3], *a.fields['c'].shape) == (0.123, 16, 16, 16)

# test centers of grains