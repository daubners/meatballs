import numpy as np
from scipy.spatial.transform import Rotation as rot
import torch.nn.functional as F
import warnings

from evoxels.pdes import SemiLinearODE
from evoxels.solvers import TimeDependentSolver
from evoxels.timesteppers import PseudoSpectralIMEX

from .fields import VoxelizedMeatball


class PITTProblem(SemiLinearODE):
    """PITT's anisotropic diffusion equation on an evoxels torch grid."""

    def __init__(self, vg, data, diffusivity=None, size=4, sigma=1.0, A=0.25):
        self.vg = vg
        self.data = data
        self.bc = "fully_periodic"
        self.initialize_boundary_conditions()
        self._fourier_symbol = -A * self.k_squared()
        self.grains = vg.to_backend(data.fields["grains"])

        if diffusivity is None:
            self.D_material = np.eye(3)
        elif isinstance(diffusivity, (list, tuple)) and len(diffusivity) == 3:
            self.D_material = np.diag(diffusivity)
        else:
            raise ValueError("Crystal diffusivity must be given as list [D_a, D_b, D_c]")

        if data.angles is None:
            raise ValueError("VoxelizedMeatball object has no grain orientations! Create orientations before trying to simulate.")
        if size % 2:
            warnings.warn("Kernel size must be even number")

        rotations = rot.from_euler("ZXZ", data.angles, degrees=True).as_matrix()
        D_grains = rotations @ self.D_material @ np.swapaxes(rotations, -1, -2)
        D_components = vg.torch.tensor(
            D_grains[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)],
            dtype=vg.precision,
            device=vg.device,
        )
        labels = self.grains
        grain_ids = vg.to_backend(data.grain_ids)
        indices = vg.torch.searchsorted(grain_ids, labels)
        D_cells = vg.torch.zeros_like(self.grains)
        active = labels > 0
        pad_size = (size - 2) // 2
        kernel = self.gaussian_kernel_3d_torch(size, sigma)
        for component, name in enumerate(("Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz")):
            D_cells.zero_()
            D_cells[active] = D_components[indices[active], component]
            D_corner = F.conv3d(
                F.pad(D_cells[None, None], (pad_size,) * 6, mode="replicate"),
                kernel,
                padding="valid",
            ).squeeze(0).squeeze(0)
            setattr(self, name, D_corner)
        del D_cells, D_components, indices, active

        self.electrolyte = (self.grains == 0).float()
        self.electrolyte[self.electrolyte == 1] = float("inf")
        self.electrolyte[self.electrolyte == 0] = 1
        self.electrolyte = self.electrolyte.unsqueeze(0)

    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol

    def rhs_analytic(self, t, u):
        raise NotImplementedError("PITT's voxelwise diffusivity has no symbolic RHS.")

    def gaussian_kernel_3d_torch(self, size, sigma):
        ax = self.vg.torch.linspace(-(size // 2), size // 2, size, device=self.vg.device)
        xx, yy, zz = self.vg.torch.meshgrid(ax, ax, ax, indexing="ij")
        kernel = self.vg.torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
        return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)

    def rotate_crystal_diffusivity_to_reference(self, bunge_angles, convention='ZXZ', degrees=True):
        # In the Bunge-Euler convention, the orientation of a crystal is represented by three Euler angles:
        # φ1 (phi1), Φ (Phi), and φ2 (phi2). These angles describe the rotations needed to bring
        # a crystal from a standard reference orientation to its current orientation.
        # Here's a brief explanation of each angle:
        #  φ1 (phi1): Rotation angle about the Z-axis of the standard reference frame.
        #  Φ (Phi): Rotation angle about the X-axis of the intermediate frame obtained after the first rotation.
        #  φ2 (phi2): Rotation angle about the Z-axis of the final frame obtained after the first two rotations.

        # Thus the full rotation from the reference into the crystal system is given by
        # [a,b,c] =  q_z(phi2) * q_x(Phi) * q_z(phi1)    * [x,y,z] and reverse
        # [x,y,z] = [q_z(phi2) * q_x(Phi) * q_z(phi1)]^T * [a,b,c]

        # Note that these are passive rotations in the sense of we are going to different coordinate systems,
        # thus in the work of Bunge, q_z = [(cos sin 0),(-sin cos 0), (0 0 1)]
        # while in scipy rot.from_euler('Z', phi) yields r_z = [(cos -sin 0),(sin cos 0), (0 0 1)] = q_x^T
        # Therefore, R = q_z^T(phi1) * q_x^T(Phi) * q_z^T(phi2)
        #              = r_z(phi1)   * r_x(Phi)   * r_z(phi2)
        #              = rot.from_euler('ZXZ', [phi1,Phi,phi2], degrees=True)

        # Transformation of e.g. the material-specific diffusion tensor into the reference space is given by
        # D_xyz = R * D_abc * R^T where R = [q_z(phi2) * q_x(Phi) * q_z(phi1)]^T
        r_matrix = rot.from_euler(convention, bunge_angles, degrees=degrees).as_matrix()
        D_xyz = r_matrix @ self.D_material @ r_matrix.T
        return self.vg.torch.tensor(D_xyz, dtype=self.vg.precision, device=self.vg.device)

    def rhs(self, t, u):
        grad = self.vg.grad_x_corner(u)
        flux_x = self.Dxx * grad
        flux_y = self.Dxy * grad
        flux_z = self.Dxz * grad

        grad = self.vg.grad_y_corner(u)
        flux_x += self.Dxy * grad
        flux_y += self.Dyy * grad
        flux_z += self.Dyz * grad

        grad = self.vg.grad_z_corner(u)
        flux_x += self.Dxz * grad
        flux_y += self.Dyz * grad
        flux_z += self.Dzz * grad

        flux_x = F.pad(flux_x, (1,) * 6, mode="constant")
        flux_y = F.pad(flux_y, (1,) * 6, mode="constant")
        flux_z = F.pad(flux_z, (1,) * 6, mode="constant")

        divergence = (self.vg.grad_x_corner(flux_x) + \
                      self.vg.grad_y_corner(flux_y) + \
                      self.vg.grad_z_corner(flux_z)) / self.electrolyte
        return divergence


class PITTSolver(TimeDependentSolver):
    """Solve PITT concentration evolution with evoxels' IMEX timestepper."""

    def __init__(self, data: VoxelizedMeatball, diffusivity=None, size=4, sigma=1.0, device="cuda", timestepper=PseudoSpectralIMEX, interpolation="arithmetic"):
        if interpolation != "arithmetic":
            raise ValueError("Only arithmetic interpolation is implemented.")
        self.data = data
        data.add_field("concentration")
        self._problem_kwargs = {
            "data": data,
            "diffusivity": diffusivity,
            "size": size,
            "sigma": sigma,
        }
        super().__init__(
            vf=data,
            fieldnames="concentration",
            backend="torch",
            problem_cls=PITTProblem,
            timestepper_cls=timestepper,
            device=device,
        )

    def solve(self, time_increment=0.01, frames=10, max_iters=1000, bc=1, A=0.25,
              verbose=True, vtk_out=False):
        self.data.fields["concentration"][self.data.fields["grains"] == 0] = bc
        self._problem_kwargs["A"] = A
        return super().solve(
            time_increment=time_increment,
            frames=frames,
            max_iters=max_iters,
            problem_kwargs=self._problem_kwargs,
            jit=False,
            verbose=verbose,
            vtk_out=vtk_out,
            colormap="turbo",
        )
