from IPython.display import clear_output
# import matplotlib.pyplot as plt
import numpy as np
# import sys
from timeit import default_timer as timer
# import psutil
from scipy.spatial.transform import Rotation as rot
import torch
# import torch.fft as fft
import torch.nn.functional as F
import warnings
from .fields import VoxelizedMeatball

class PITTSolver:
    def __init__(self, data: VoxelizedMeatball, diffusivity=None, size=4, sigma=1.0, device='cuda'):
        """
        Solves concentration evolution for anisotropic diffusion in polycrystalline agglomerate.

        Args:
            data (VoxelizedMeatball): The voxel agglomerate object containing spatial and field information.
            device (str): The device to perform computations ('cpu' or 'cuda').
        """
        self.data = data
        data.add_field('concentration')
        self.device = torch.device(device)
        # check device is available
        if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            warnings.warn("CUDA not available, defaulting device to cpu. To avoid this warning, set device=torch.device('cpu')")
        self.precision = torch.float32 # torch.float64

        self.grains = torch.tensor(data.fields['grains'], dtype=self.precision, device=self.device)
        self.mask = (self.grains == 0).float()
        self.mask = self.mask.unsqueeze(0)
        self.c = torch.zeros_like(self.grains)

        if not diffusivity:
            self.D_material = np.eye(3)
        else:
            if not isinstance(diffusivity, (list, tuple)) or len(diffusivity) != 3:
                raise ValueError("Crystal diffusivity must be given as list [D_a, D_b, D_c]")
            self.D_material = np.eye(3)
            self.D_material[0,0] = diffusivity[0]
            self.D_material[1,1] = diffusivity[1]
            self.D_material[2,2] = diffusivity[2]

        # Diffusivities and fluxes are defined at cell corners
        # TODO: refactor this lengthy bit
        self.Dxx = torch.zeros(self.data.Nx-1, self.data.Ny-1, self.data.Nz-1, dtype=self.precision, device=self.device)
        self.Dxy = torch.zeros(self.data.Nx-1, self.data.Ny-1, self.data.Nz-1, dtype=self.precision, device=self.device)
        self.Dxz = torch.zeros(self.data.Nx-1, self.data.Ny-1, self.data.Nz-1, dtype=self.precision, device=self.device)
        self.Dyy = torch.zeros(self.data.Nx-1, self.data.Ny-1, self.data.Nz-1, dtype=self.precision, device=self.device)
        self.Dyz = torch.zeros(self.data.Nx-1, self.data.Ny-1, self.data.Nz-1, dtype=self.precision, device=self.device)
        self.Dzz = torch.zeros(self.data.Nx-1, self.data.Ny-1, self.data.Nz-1, dtype=self.precision, device=self.device)

        if size % 2 > 0:
            warnings.warn("Kernel size must be even number")
        pad_size = int((size-2)/2)
        gaussian = self.gaussian_kernel_3d_torch(size=size, sigma=sigma)
        
        if data.angles is None:
            raise ValueError("VoxelizedMeatball object has no grain orientations! Create orientations before trying to simulate.")

        for i, label in enumerate(data.grain_ids):
            # sub_tensor, indices = self.crop_area_of_interest_torch(tensor, label)
            # Create binary mask for the label within the slice
            # mask = (sub_tensor == label).float()
            mask = (self.grains == label).float()
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = F.pad(mask, (pad_size,pad_size,pad_size,pad_size,pad_size,pad_size), mode='replicate')
            mask = F.conv3d(mask, gaussian, padding='valid')
            mask = mask.squeeze()
            D_grain = self.rotate_crystal_diffusivity_to_reference(data.angles[i], convention='ZXZ', degrees=True)
            self.Dxx += mask*D_grain[0,0]
            self.Dxy += mask*D_grain[0,1]
            self.Dxy += mask*D_grain[0,2]
            self.Dyy += mask*D_grain[1,1]
            self.Dyz += mask*D_grain[1,2]
            self.Dzz += mask*D_grain[2,2]

        # TODO: correction at interface
        # self.Dxx = Dxx/ phi_active
        
    def print_memory_stats(self, start, end, iters):
        print(f'Wall time: {np.around(end - start, 4)} s ({np.around((end - start)/iters, 4)} s/iter)')
        if self.device.type == 'cuda':
            print(f"GPU-RAM currently allocated {torch.cuda.memory_allocated(device=self.device) / 1e6:.2f} MB ({torch.cuda.memory_reserved(device=self.device) / 1e6:.2f} MB reserved)")
            print(f"GPU-RAM maximally allocated {torch.cuda.max_memory_allocated(device=self.device) / 1e6:.2f} MB ({torch.cuda.max_memory_reserved(device=self.device) / 1e6:.2f} MB reserved)")
        elif self.device.type == 'cpu':
            memory_info = psutil.virtual_memory()
            print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
            print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
            print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")

    def handle_outputs(self, vtk_out, verbose, slice=0):
        self.data.fields['concentration'] = self.c.squeeze().cpu().numpy()
        if np.isnan(self.data.fields['concentration']).any():
            print(f"NaN detected in frame {self.frame} at time {self.time}. Aborting simulation.")
            sys.exit(1)  # Exit the program with an error status
        if vtk_out:
            filename = f"PITT_c_{self.frame:03d}.vtk"
            self.data.export_to_vtk(filename=filename, field_names=['concentration'])
        if verbose == 'plot':
            clear_output(wait=True)
            self.data.plot_slice('concentration', slice, time=self.time)

    def gaussian_kernel_3d_torch(self, size=4, sigma=1.0):
        """Creates a 3D Gaussian kernel using PyTorch"""
        ax = torch.linspace(-(size // 2), size // 2, size)
        xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")

        # Calculate Gaussian function for each point in the grid
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        kernel = kernel.to(self.device)
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def crop_area_of_interest_torch(self, tensor, labels):
        indices = torch.nonzero(torch.isin(tensor, labels), as_tuple=True)
        min_idx = [torch.min(idx).item() for idx in indices]
        max_idx = [torch.max(idx).item() for idx in indices]

        # Slice the tensor to the bounding box
        # Make sure to stay inside the bounds of total array
        box_idx = (slice(max(min_idx[0] - 3, 0), min(max_idx[0] + 4, tensor.shape[0])),
                  slice(max(min_idx[1] - 3, 0), min(max_idx[1] + 4, tensor.shape[1])),
                  slice(max(min_idx[2] - 3, 0), min(max_idx[2] + 4, tensor.shape[2]))
                 )
        sub_tensor = tensor[box_idx]
        return sub_tensor, box_idx
    
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
        return torch.tensor(D_xyz, dtype=self.precision, device=self.device)

    def calc_grad_x(self, tensor):
        return (  tensor[:, 1: , 1: , 1:] + tensor[:, 1: , 1: , :-1] \
                + tensor[:, 1: , :-1, 1:] + tensor[:, 1: , :-1, :-1] \
                - tensor[:, :-1, 1: , 1:] - tensor[:, :-1, 1: , :-1] \
                - tensor[:, :-1, :-1, 1:] - tensor[:, :-1, :-1, :-1] ) / 4
    
    def calc_grad_y(self, tensor):
        return (  tensor[:, 1: , 1: , 1:] + tensor[:, 1: , 1: , :-1] \
                + tensor[:, :-1, 1: , 1:] + tensor[:, :-1, 1: , :-1] \
                - tensor[:, 1: , :-1, 1:] - tensor[:, 1: , :-1, :-1] \
                - tensor[:, :-1, :-1, 1:] - tensor[:, :-1, :-1, :-1] ) / 4

    def calc_grad_z(self, tensor):
        return (  tensor[:, 1: , 1:, 1: ] + tensor[:, 1: , :-1 , 1:] \
                + tensor[:, :-1, 1:, 1: ] + tensor[:, :-1, :-1 , 1:] \
                - tensor[:, 1: , 1:, :-1] - tensor[:, 1: , :-1, :-1] \
                - tensor[:, :-1, 1:, :-1] - tensor[:, :-1, :-1, :-1] ) / 4

    def solve(self, time_increment=0.01, frames=10, max_iters=1000, bc=1, verbose=True, vtk_out=False):
        """
        Solves concentration evolution using explicit timestepping.
        """
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=self.device)
        self.n_out = int(max_iters/frames)
        self.frame = 0
        self.time = 0
        self.c = self.c.unsqueeze(0)
        slice = int(self.c.shape[-1]/2)
        with torch.no_grad():
            start = timer()
            for i in range(max_iters):
                if i % self.n_out == 0:
                    self.time = i*time_increment
                    self.handle_outputs(vtk_out, verbose, slice=slice)
                    self.frame += 1

                # Apply electrolyte boundary conditions
                # Set c=bc in all electrolyte cells
                self.c *= (1-self.mask)
                self.c += self.mask * bc

                grad = self.calc_grad_x(self.c)
                flux_x = self.Dxx*grad
                flux_y = self.Dxy*grad
                flux_z = self.Dxz*grad

                grad = self.calc_grad_y(self.c)
                flux_x += self.Dxy*grad
                flux_y += self.Dyy*grad
                # flux_z += self.Dyz*grad

                grad = self.calc_grad_z(self.c)            
                flux_x += self.Dxz*grad
                # flux_y += self.Dyz*grad
                flux_z += self.Dzz*grad

                # Add Neumann boundary conditions
                flux_x = F.pad(flux_x, (1,1,1,1,1,1), mode='constant')
                flux_y = F.pad(flux_y, (1,1,1,1,1,1), mode='constant')
                flux_z = F.pad(flux_z, (1,1,1,1,1,1), mode='constant')

                self.c += time_increment * (self.calc_grad_x(flux_x) + \
                                            self.calc_grad_y(flux_y) + \
                                            self.calc_grad_z(flux_z) )

            end = timer()
            self.time = max_iters*time_increment
            self.handle_outputs(vtk_out, verbose, slice=slice)
            if verbose:
                self.print_memory_stats(start, end, max_iters)