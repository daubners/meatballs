# In a world of cubes and blocks,
# Where reality takes voxel knocks,
# Every shape and form we see,
# Is a pixelated mystery.

# Mountains rise in jagged peaks,
# Rivers flow in blocky streaks.
# So embrace the charm of this edgy place,
# Where every voxel finds its space

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pyvista as pv
import warnings
from .utils import create_NMC_agglomerate

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

    def create_random_agglomerate(self, radius: int, num_seeds: int):
        grains, angles = create_NMC_agglomerate([radius], [num_seeds])
        self.__initialize_grain_map(grains)
        self.angles = angles

    def create_structured_agglomerate(self, radii, num_seeds):
        grains, angles = create_NMC_agglomerate(radii, num_seeds)
        self.__initialize_grain_map(grains)
        self.angles = angles

    def add_random_orientations(self):
        if self.angles is not None:
            warnings.warn("Previous angles will be over-written!")
        if self.grain_ids:
            print("Generating random orientations...")
            self.angles = 180*np.random.rand(len(self.grain_ids), 3)
        else:
            raise ValueError("Cannot add orientations to non-existing grains. Create grains first.")

    def guess_grain_orientation_from_shape(self):
        print("Estimating grain orientations from crystal shapes!")
        # Assuming that the orientation of the smalles half-axis in PCA space
        # coresponds to the c-axis (because that is the slowest growing crystal axis).
        # angle_list = self.guess_grain_orientation_from_shape()

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

    def export_to_vtk(self, filename="output.vtk", field_names=None):
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
            end1, end2 = self.spacing[1], self.spacing[2]
            label1, label2 = ['Y', 'Z']
        elif direction == 'y':
            axes = (1,0,2)
            end1, end2 = self.spacing[0], self.spacing[2]
            label1, label2 = ['X', 'Z']
        elif direction == 'z':
            axes = (2,0,1)
            end1, end2 = self.spacing[0], self.spacing[1]
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