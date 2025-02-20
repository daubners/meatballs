import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as rot
import warnings
from .fields import VoxelizedMeatball

def evaluate_grain_alignment_dot_product(data: VoxelizedMeatball):
    """
    Evaluate grain aligment by computing the dot product between:
     - unit vector from the agglomerate center to each grain center, and 
     - grains' c-axis (extracted from its Euler angles).
   
    Returns:
        dot_products (dict): A dictionary mapping each grain label to the dot product value.
                             The dot product is computed as:
                               dot( unit_vector( (grain_center - agglomerate_center) ),
                                    c_axis_of_grain )
    """
    # Compute agglomerate center based on the mean of all nonzero voxel indices.
    nonzero_voxels = np.argwhere(data.fields['grains'] != 0)
    mean_voxel = np.mean(nonzero_voxels, axis=0)
    agglomerate_center = data.origin + mean_voxel * data.spacing

    grain_centers = data.compute_grain_centers(data.fields['grains'])
    
    dot_products = np.zeros((len(data.grain_ids)))

    for i, label in enumerate(data.grain_ids):
        grain_center = grain_centers[i]
        # Compute vector from agglomerate center to grain center and normalize it.
        v = grain_center - agglomerate_center
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            # Avoid division by zero – if grain center = agglomerate center.
            unit_v = np.zeros_like(v)
        else:
            unit_v = v / norm_v
        
        R_matrix = rot.from_euler('ZXZ', data.angles[i], degrees=True).as_matrix()
        c_axis = R_matrix @ np.array([0, 0, 1])

        dp = np.dot(unit_v, c_axis)
        dot_products[i] = np.abs(dp)

    return dot_products

def plot_grain_alignment(data: VoxelizedMeatball):
    """
    Plots the dot product values versus grain IDs.
    
    Args:
        dot_products (dict): A dictionary mapping grain IDs to dot product values.
    """
    dp_values = evaluate_grain_alignment_dot_product(data)
    
    plt.figure(figsize=(8, 5))
    plt.plot(data.grain_ids, dp_values, 'o', color='blue')
    plt.xlabel("Grain ID")
    plt.ylabel("Dot Product")
    plt.title("Grain alignment given as dot product between radial direction and grain c-axis")
    plt.grid(True)
    plt.show()