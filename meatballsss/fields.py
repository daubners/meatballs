# In a world of cubes and blocks,
# Where reality takes voxel knocks,
# Every shape and form we see,
# Is a pixelated mystery.

# Mountains rise in jagged peaks,
# Rivers flow in blocky streaks.
# So embrace the charm of this edgy place,
# Where every voxel finds its space

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import warnings

from IPython.display import clear_output
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse, Polygon
from scipy.spatial.transform import Rotation as rot
from skimage.segmentation import find_boundaries
from sklearn.decomposition import PCA
from .utils import create_NMC_agglomerate, normalize_angles

class VoxelizedMeatball:
    """
    Represents a 3D NMC agglomerate on a voxel grid including grain orientations.
    This class acts as a container to manage properties of an agglomerate, such as
    a labelled grain map and corresponding angle list together with grid information.
    Further functionality includes conversion to an angle map (for crystalGAN),
    generation of synthetic grain structures, generation of grain orientations,
    visualization and vtk export for faster solver development.
    The voxel representation assumes a cell-center convention.

    Attributes:
        Nx, Ny, Nz: Number of voxels along the x, y and z-axis.
        domain_size (tuple): Length of physical domain (Lx, Ly, Lz)
        spacing (tuple): Grid spacing along each axis (dx, dy, dz).
        origin (tuple): Position of lower left corner (for vtk export)
        fields (dict): Dictionary to store named 3D fields.
    """

    def __init__(self, grain_map = None, angle_list = None, spacing = None):
        """
        Initializes the voxel agglomerate based on given grain map or empty.
        Parameters:
            grain_map (np.ndarray): A 3D numpy array of shape (Nx, Ny, Nz) where each voxel is labelled with a unique grain ID.
                                    Electrolyte/ pore must have label '0'.
            angle_list (np.ndarray): List of orientations corresponding to labelled grains.
            spacing (tuple or list or np.ndarray): The voxel spacing in the x, y, z directions (dx, dy, dz).
        Raises:
            ValueError: If spacing is not a list or tuple with three elements or contains non-numeric values.
            Warning: If spacings differ significantly, a warning is issued.
        """
        if spacing:
            if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
                raise ValueError("spacing must be a list or tuple with three elements (dx, dy, dz)")
            if not all(isinstance(x, (int, float)) for x in spacing):
                raise ValueError("All elements in spacing must be integers or floats")
            self.spacing = spacing
        else:
            self.spacing = (1,1,1)
        if (np.max(self.spacing)/np.min(self.spacing) > 10):
            warnings.warn("Simulations become very questionable for largely different spacings e.g. dz >> dx.")
        self.origin = (self.spacing[0]/2, self.spacing[1]/2, self.spacing[2]/2)

        self.fields = {}
        self.angles = None
        self.grain_ids = None
        if grain_map is not None:
            if not isinstance(grain_map, np.ndarray):
                raise TypeError("Error: grain_map must be a NumPy array!")
            if grain_map.ndim == 2:
                # Convert 2D to pseudo-3D by stacking along the third axis
                grain_map = np.stack([grain_map, grain_map], axis=-1)
            elif grain_map.ndim != 3:
                raise ValueError("Error: grain_map must be either 2D or 3D!")

            self.__initialize_grain_map(grain_map)
            if angle_list is not None:
                # TODO: should the angle_list be a dictionary to be less error-prone in case labels are not 1,2,3,...?
                if isinstance(angle_list, np.ndarray):
                    if angle_list.ndim != 2:
                        raise ValueError(f"angle_list must be a 2D NumPy array of shape [{len(self.grain_ids)}, 3]")
                    dim1, dim2 = angle_list.shape
                    if dim1 == len(self.grain_ids) and dim2 == 3:
                        self.angles = angle_list
                    else:
                        raise ValueError(f"angle_list must have shape [{len(self.grain_ids)}, 3], but got {angle_list.shape}")
                else:
                    raise TypeError("angle_list must be a NumPy array!")
        else:
            if angle_list is not None:
                raise ValueError("Angle list but no corresponding grain map has been given!")
            warnings.warn("Empty initialization. Please generate a grain structure!")

    def __initialize_grain_map(self, grain_map):
        if 'grains' in self.fields:
            warnings.warn("Previous grain map will be over-written!")
        self.fields['grains'] = grain_map
        self.Nx, self.Ny, self.Nz = grain_map.shape
        self.domain_size = (self.spacing[0]*self.Nx, self.spacing[1]*self.Ny, self.spacing[2]*self.Nz)

        labels = np.unique(grain_map)
        if np.min(labels) > 0:
            warnings.warn("Electrolyte is assumed to have label 0, but no pixel with label 0 was found!")
        self.grain_ids = labels[labels > 0]

    def compute_grain_centers(self, field):
        """
        Compute the center of mass for each grain in a labelled pixel/voxel grid.
        Works for both 2D slices and 3D grids.
        Returns:
            np.ndarray: An array of shape (num_grains, D) where D is the number of dimensions,
                    containing the center of mass for each grain.
        """ 
        dim = field.ndim
        grain_centers = np.zeros((len(self.grain_ids), dim))
        
        for i, label in enumerate(self.grain_ids):
            # Find voxel indices (i, j, k) where the voxel belongs to the current grain.
            indices = np.argwhere(field == label)  # shape: (N_points, 3)
            mean_indices = np.mean(indices, axis=0)  # [mean_x, mean_y, mean_z]
            grain_centers[i] = self.origin + mean_indices * self.spacing

        return grain_centers

    def create_random_agglomerate(self, radius: int, num_seeds: int):
        grains, angles = create_NMC_agglomerate([radius], [num_seeds])
        self.__initialize_grain_map(grains)
        self.angles = angles

    def create_structured_agglomerate(self, radii, num_seeds):
        grains, angles = create_NMC_agglomerate(radii, num_seeds)
        self.__initialize_grain_map(grains)
        self.angles = angles

    def color_array_to_grain_map(self, img_array, tolerance=0, normalize_angles=None):
        """
        Convert a segmented color image to a grainmap with unique IDs.
        If the image is 2D (i.e. no explicit depth dimension), a pseudo-3D grain map is created
        by stacking the 2D label map along a new axis.
        
        Parameters:
            img_array (np.ndarray): Input image array.
            tolerance (int): Tolerance for color matching (currently not implemented)
        """
        # Remove the alpha channel if present.
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        
        # Check if the image is 2D or 3D based on the number of spatial dimensions.
        if img_array.ndim == 3:  # 2D image with shape (H, W, 3)
            spatial_shape = img_array.shape[:2]
        elif img_array.ndim == 4:  # 3D image with shape (X, Y, Z, 3)
            spatial_shape = img_array.shape[:3]
        else:
            raise ValueError("Unsupported input dimensions. Expecting a 2D or 3D RGB image.")
        # Reshape to a list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # If a tolerance is set, additional post-processing would be required here.
        # For simplicity, we assume tolerance==0 (exact matching).
        # TODO: mechanism that black is always background label 0
        _, inverse_indices = np.unique(pixels, axis=0, return_inverse=True)
        grains = inverse_indices.reshape(spatial_shape)

        if grains.ndim == 2:
            self.__initialize_grain_map(np.stack([grains, grains], axis=-1))
        else:
            self.__initialize_grain_map(grains)

        # TODO: this now assumes colorvalues between 0 and 1 for each channel
        # TODO: make sure normalize_angles is list of three angles
        if normalize_angles is not None:
            self.angles = np.zeros((len(self.grain_ids), 3))
            normalize_angles = np.array(normalize_angles)
            for i, label in enumerate(self.grain_ids):
                self.angles[i] = normalize_angles*img_array[grains==label][0]
        else:
            self.angles = None

    def add_random_orientations(self, angle_range=[180,180,120]):
        if self.angles is not None:
            warnings.warn("Previous angles will be over-written!")
        if self.grain_ids is not None:
            print("Generating random orientations...")
            # Per default last angle is defined between 0 and 120 degrees (hexagonal unit cell)
            self.angles = np.column_stack((angle_range[0]*np.random.rand(len(self.grain_ids), 1), \
                                           angle_range[1]*np.random.rand(len(self.grain_ids), 1), \
                                           angle_range[2]*np.random.rand(len(self.grain_ids), 1)))
        else:
            raise ValueError("Cannot add orientations to non-existing grains. Create grains first.")

    def guess_grain_orientation_from_shape(self, voxel_threshold=3):
        """
        Compute the orientation of each grain in a 3D labelled voxel grid based on its shape,
        and return the corresponding Euler angles [φ₁, Φ, φ₂] in the Bunge ZXZ convention.
        If a grain has fewer than voxel_threshold voxels, a rotation of [0,0,0] is returned per default.

        For each grain (with label > 0):
        - PCA is performed on the voxel coordinates.
        - Eigenvector with largest eigenvalue (longest dimension) is taken as the a‑axis.
        - Eigenvector with smallest eigenvalue (shortest dimension) is taken as the c‑axis
          because that is the slowest growing crystal axis.
        - The b‑axis is implicitly given as the second principal component.
        """
        if self.angles is not None:
            warnings.warn("Previous angles will be over-written!")
        if self.grain_ids is not None:
            print("Estimating grain orientations from crystal shapes!")
            bunge_angles = np.zeros((len(self.grain_ids), 3))
            
            for i, label in enumerate(self.grain_ids):
                grain_mask = (self.fields['grains'] == label)
                points = np.column_stack(np.where(grain_mask)) # shape (n_points, 3)
                
                # Check if there are enough points to perform a meaningful 3D PCA.
                if points.shape[0] < voxel_threshold:
                    continue
                
                # Perform PCA on the voxel coordinates.
                pca = PCA(n_components=3)
                pca.fit(points)
                # PCA components are sorted in order of descending variance:
                # The first component is the a-axis (longest dimension),
                # the third is the c-axis (shortest dimension).
                # Transpose of pca.components_ corresponds to rotation matrix
                R_matrix = pca.components_.T
                # Make sure this is a right-handed rotation system
                if np.linalg.det(R_matrix) < 0:
                    R_matrix[:, 2] *= -1
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
                    bunge_angles[i] = rot.from_matrix(R_matrix).as_euler('ZXZ', degrees=True)
            bunge_angles = normalize_angles(bunge_angles)
            self.angles = bunge_angles
        else:
            raise ValueError("Cannot add orientations to non-existing grains. Create grains first.")

    def relabel_random_order(self):
        old_labels = np.unique(self.fields['grains'])
        new_labels = np.arange(len(old_labels))
        # Zero should be kept where it is
        np.random.shuffle(new_labels[1:])

        # Create a mapping from old labels to new shuffled labels
        label_mapping = dict(zip(old_labels, new_labels))

        # Vectorized relabeling using np.vectorize for efficiency
        relabel_function = np.vectorize(lambda x: label_mapping[x])
        self.fields['grains'] = relabel_function(self.fields['grains'])

        if self.angles is not None:
            shuffled_angles = self.angles[np.argsort(new_labels[1:])]
            self.angles = shuffled_angles

    def add_field(self, name: str, array=None):
        """
        Adds a field to the voxel grid.

        Args:
            name (str): Name of the field.
            array (numpy.ndarray, optional): 3D array to initialize the field. If None, initializes with zeros.

        Raises:
            ValueError: If the provided array does not match the voxel grid dimensions.
            TypeError: If the provided array is not a numpy array.
        """
        if array is not None:
            if isinstance(array, np.ndarray):
                if array.shape == (self.Nx, self.Ny, self.Nz):
                    self.fields[name] = array
                else:
                    raise ValueError(f"The provided array must have the shape ({self.Nx}, {self.Ny}, {self.Nz}).")
            else:
                raise TypeError("The provided array must be a numpy array.")
        else:
            self.fields[name] = np.zeros((self.Nx, self.Ny, self.Nz))

    def generate_color_map(self, normalize_angles=[180,180,120]):
        if 'grains' in self.fields:
            if self.angles is not None:
                print("Generating colormap from grain_map and angle_list.")
                colormap = np.zeros((self.Nx, self.Ny, self.Nz, 3))
                for i, label in enumerate(self.grain_ids):
                    colormap[self.fields['grains']==label] = self.angles[i]/normalize_angles
                return colormap
            else:
                raise ValueError("Create orientations before trying to generate color map.")
        else:
            raise ValueError("Add grain_map before trying to generate color map.")

    def export_orientations_to_vtk(self, filename="orientations.vtk"):
        """
        Exports orientations as vector data to a VTK file for visualization (e.g. VisIt or ParaView).
        Args:
            filename (str): Name of the output VTK file.
        """
        centers = self.compute_grain_centers(self.fields['grains'])
        a_axis = np.zeros((len(self.grain_ids), 3))
        b_axis = np.zeros((len(self.grain_ids), 3))
        c_axis = np.zeros((len(self.grain_ids), 3))
        for i, label in enumerate(self.grain_ids):
            Q = rot.from_euler('ZXZ', self.angles[i], degrees=True).as_matrix()
            a_axis[i] = Q @ np.array([1, 0, 0])
            b_axis[i] = Q @ np.array([0, 1, 0])
            c_axis[i] = Q @ np.array([0, 0, 1])

        point_cloud = pv.PolyData(centers)
        point_cloud["a_axis"] = a_axis
        point_cloud["b_axis"] = b_axis
        point_cloud["c_axis"] = c_axis
        point_cloud.save(filename)

    def export_fields_to_vtk(self, filename="fields.vtk", field_names=None):
        """
        Exports fields to a VTK file for visualization (e.g. VisIt or ParaView).

        Args:
            filename (str): Name of the output VTK file.
            field_names (list, optional): List of field names to export. Exports all fields if None.
        """
        grid = pv.ImageData()
        grid.dimensions = (self.Nx + 1, self.Ny + 1, self.Nz + 1)
        grid.spacing = self.spacing
        grid.origin = (self.origin[0] - self.spacing[0]/2, self.origin[1] - self.spacing[1]/2, self.origin[2] - self.spacing[2]/2)

        names = field_names if field_names else list(self.fields.keys())
        for name in names:
            grid.cell_data[name] = self.fields[name].flatten(order="F")  # Fortran order flattening
        grid.save(filename)

    def plot_slice(self, fieldname, slice_index, direction='z', time=None, colormap='viridis'):
        """
        Plots a 2D slice of a field along a specified direction.

        Args:
            fieldname (str): Name of the field to plot.
            slice_index (int): Index of the slice to plot.
            direction (str): Normal direction of the slice ('x', 'y', or 'z').
            dpi (int): Resolution of the plot.
            colormap (str): Colormap to use for the plot.

        Raises:
            ValueError: If an invalid direction is provided.
        """
        # Colormaps
        # linear: viridis, Greys
        # diverging: seismic
        # levels: tab20, flag
        # gradual: turbo
        if direction == 'x':
            slice = np.s_[slice_index,:,:]
            end1, end2 = self.domain_size[1], self.domain_size[2]
            label1, label2 = ['Y', 'Z']
        elif direction == 'y':
            slice = np.s_[:,slice_index,:]
            end1, end2 = self.domain_size[0], self.domain_size[2]
            label1, label2 = ['X', 'Z']
        elif direction == 'z':
            slice = np.s_[:,:,slice_index]
            end1, end2 = self.domain_size[0], self.domain_size[1]
            label1, label2 = ['X', 'Y']
        else:
            raise ValueError("Given direction must be x, y or z")

        plt.figure()
        im = plt.imshow(self.fields[fieldname][slice].T, cmap=colormap, origin='lower', extent=[0, end1, 0, end2])
        plt.colorbar(im)
        plt.xlabel(label1)
        plt.ylabel(label2)
        if time:
            plt.title(f'Slice {slice_index} of {fieldname} in {direction} at time {time}')
        else:
            plt.title(f'Slice {slice_index} of {fieldname} in {direction}')
        plt.show()

    def plot_field_interactive(self, fieldname, direction='x', colormap='viridis'):
        """
        Creates an interactive plot for exploring slices of a 3D field.

        Args:
            fieldname (str): Name of the field to plot.
            direction (str): Direction of slicing ('x', 'y', or 'z').
            dpi (int): Resolution of the plot.
            colormap (str): Colormap to use for the plot.

        Raises:
            ValueError: If an invalid direction is provided.
        """
        if direction == 'x':
            axes = (0,1,2)
            end1, end2 = self.domain_size[1], self.domain_size[2]
            label1, label2 = ['Y', 'Z']
        elif direction == 'y':
            axes = (1,0,2)
            end1, end2 = self.domain_size[0], self.domain_size[2]
            label1, label2 = ['X', 'Z']
        elif direction == 'z':
            axes = (2,0,1)
            end1, end2 = self.domain_size[0], self.domain_size[1]
            label1, label2 = ['X', 'Y']
        else:
            raise ValueError("Given direction must be x, y or z")

        field = np.transpose(self.fields[fieldname], axes)
        max_id = np.max(np.unique(field))
        fig, ax = plt.subplots()
        im = ax.imshow(field[0].T, cmap=colormap, origin='lower', extent=[0, end1, 0, end2], vmin=0, vmax=max_id)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_title(f'Slice 0 in {direction}-direction of {fieldname}')
        plt.colorbar(im, ax=ax)

        # Add a slider for changing timeframes
        position = plt.axes([0.2, 0.0, 0.6, 0.02])
        ax_slider = Slider(position, 'Slice', 0, field.shape[0]-1, valinit=0, valstep=1)

        def update(val):
            slice_idx = int(ax_slider.val)
            im.set_array(field[slice_idx].T)
            ax.set_title(f'Slice {slice_idx} in ' + direction + '-direction of ' + fieldname)
            fig.canvas.draw_idle()

        ax_slider.on_changed(update)
        return ax_slider
    
    def plot_slice_with_orientations(self, slice_index, direction='z', show_ids=False, colormap='viridis'):
        """
        Plots a 2D slice of a labelled grain map and overlays grain orientation arrows.
        
        For each grain that is visible in the slice, the function computes the 2D center of the grain
        (from pixels in the slice) and overlays an arrow representing the projection of the grain's 
        c-axis (derived from the provided Euler angles using the Bunge ZXZ convention) onto the slice plane.
        
        Parameters:
            slice_index (int): The index of the slice to plot.
            direction (str): The normal direction of the slice ('x', 'y', or 'z').
            show_ids (bool): If True, overlay the grain IDs at the computed grain centers.
            colormap (str): Colormap used for displaying the slice.
        """
        if direction == 'x':
            slice = np.s_[slice_index, :, :]
            end1, end2 = self.domain_size[1], self.domain_size[2]
            label1, label2 = 'Y', 'Z'
        elif direction == 'y':
            slice = np.s_[:, slice_index, :]
            end1, end2 = self.domain_size[0], self.domain_size[2]
            label1, label2 = 'X', 'Z'
        elif direction == 'z':
            slice = np.s_[:, :, slice_index]
            end1, end2 = self.domain_size[0], self.domain_size[1]
            label1, label2 = 'X', 'Y'
        else:
            raise ValueError("Direction must be 'x', 'y', or 'z'.")

        # Extract the 2D slice.
        slice_img = self.fields['grains'][slice]
        visible_grains = np.unique(slice_img)
        visible_grains = visible_grains[visible_grains != 0]
        masked_slice_img = np.ma.masked_equal(slice_img, 0)
        cmap = plt.get_cmap(colormap).copy()
        cmap.set_bad(color='black')

        boundaries = find_boundaries(slice_img, mode='inner')

        plt.figure(figsize=(10, 10))
        im = plt.imshow(masked_slice_img.T, cmap=cmap, origin='lower', extent=[0, end1, 0, end2])
        plt.contour(boundaries.T, colors='gray', linewidths=0.5)
        plt.colorbar(im)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title(f'Slice {slice_index} along {direction}')
        ax = plt.gca()
        label_to_index = {int(label): i for i, label in enumerate(self.grain_ids)}
        scale = (np.max([self.Nx, self.Ny, self.Nz]) / 20)

        for grain_id in visible_grains:
            indices = np.argwhere(slice_img == grain_id)
            center = indices.mean(axis=0)  # center[0]: row, center[1]: column
            
            R_matrix = rot.from_euler('ZXZ', self.angles[label_to_index[grain_id]], degrees=True).as_matrix()
            hex_angles = np.deg2rad(np.arange(0, 360, 60))  # 0, 60, 120, ..., 300 degrees.
            # The unit vectors in 3D (with zero z-component)
            v = np.column_stack((np.cos(hex_angles), np.sin(hex_angles), np.zeros_like(hex_angles)))  # shape (6,3)
            corners_3d = (R_matrix @ v.T).T
            hex_vertices = corners_3d[:, :2]
            hex_vertices = 0.5 * scale * hex_vertices + np.array(center)
            hex_patch = Polygon(hex_vertices, closed=True, edgecolor="black", facecolor='white', lw=1, alpha=0.5)
            ax.add_patch(hex_patch)
            
            # Compute the 3D ellipsoid axes in the global frame
            matrices = [np.diag([1, 0.05, 0.05]), \
                rot.from_euler('z', 120, degrees=True).as_matrix() @ np.diag([1, 0.05, 0.05]), \
                rot.from_euler('z', -120, degrees=True).as_matrix() @ np.diag([1, 0.05, 0.05])]
            ellipsoids = [R_matrix @ M for M in matrices]

            for ellipsoid in ellipsoids:
                if direction == 'x':
                    ellipse = np.array([[0, 1, 0], [0, 0, 1]]) @ ellipsoid
                elif direction == 'y':
                    ellipse = np.array([[1, 0, 0], [0, 0, 1]]) @ ellipsoid
                elif direction == 'z':
                    ellipse = np.array([[1, 0, 0], [0, 1, 0]]) @ ellipsoid

                U, S, _ = np.linalg.svd(ellipse)
                angle_proj = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

                # Create the ellipse patch representing the projected ellipsoid.
                width = S[0] * scale  # full length (major axis)
                height = S[1] * scale  # full length (minor axis)
                patch = Ellipse((center[0], center[1]), width=width, height=height,
                                angle=angle_proj, edgecolor="gray", facecolor="red", lw=1,\
                                alpha=0.5)
                ax.add_patch(patch)

            # Optionally, overlay the grain ID.
            if show_ids:
                plt.text(center[0], center[1], str(int(grain_id)), color='blue', fontsize=12,
                        ha='center', va='center')
        
        plt.axis('off')
        plt.show()