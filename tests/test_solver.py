"""Tests for the PITT solver's evoxels integration."""

import pytest

import numpy as np

from evoxels.timesteppers import ForwardEuler, PseudoSpectralIMEX
from meatballs import PITTSolver, VoxelizedMeatball
from meatballs.solver import PITTProblem


@pytest.mark.parametrize("timestepper", [PseudoSpectralIMEX, ForwardEuler])
def test_pitt_uses_evoxels_imex_torch_grid(timestepper):
    grains = np.zeros((8, 8, 8), dtype=int)
    grains[2:6, 2:6, 2:6] = 1
    data = VoxelizedMeatball(grains, angle_list=np.zeros((1, 3)))
    solver = PITTSolver(data, device="cpu", timestepper=timestepper)

    solver.solve(time_increment=0.01, frames=1, max_iters=1, verbose=False)

    assert isinstance(solver.problem, PITTProblem)
    single_tensor = solver.problem.rotate_crystal_diffusivity_to_reference(data.angles[0])
    grain_angles = np.array([data.angles[0], [0, 90, 0]])
    grain_tensors = solver.problem.rotate_crystal_diffusivity_to_reference(grain_angles)
    assert single_tensor.shape == (3, 3)
    assert grain_tensors.shape == (2, 3, 3)
    assert np.allclose(single_tensor.cpu(), grain_tensors[0].cpu())
    assert solver.timestepper_cls is timestepper
    assert data.fields["concentration"].shape == (8, 8, 8)
    assert np.isfinite(data.fields["concentration"]).all()


def test_pitt_rejects_unsupported_interpolation():
    data = VoxelizedMeatball(np.zeros((2, 2, 2), dtype=int))

    with pytest.raises(ValueError, match="Only arithmetic interpolation"):
        PITTSolver(data, interpolation="harmonic")
