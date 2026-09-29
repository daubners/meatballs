import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as rot
import torch
import warnings
from .fields import VoxelizedMeatball

def evaluate_global_grain_alignment(data: VoxelizedMeatball):
    """
    Evaluate grain aligment by computing the dot product between:
     - unit vector from the agglomerate center to each grain center, and 
     - grains' c-axis (extracted from its Euler angles).
   
    Returns:
        dot_products (array): A numpy array mapping each grain label to the dot product value.
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


def evaluate_relative_grain_alignment(data: VoxelizedMeatball):
    """
    Evaluate relative grain aligment by computing the disorientation angles
    (smallest possible rotation angle out of all symmetrically equivalent misorientations)
     - a-axes (taking hexagonal symmetry into account, therefore, b-axis would be identical) 
     - c-axis (no additional symmetry)
    for each grain-pair.
   
    Returns:
        dot_products (array)
    """
    # Determine pairs of neighbouring grains
    tensor = torch.tensor(data.fields['grains'])
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    tensor = tensor.to(device)
    tensor = tensor.to(torch.int32)

    phasepairs = torch.tensor([[0,0]], device=device)
    # Check x neighbours
    neighbour_idx = torch.nonzero(tensor[:-1,:,:] != tensor[1:,:,:], as_tuple=True)
    neighbour_list = torch.stack([tensor[:-1,:,:][neighbour_idx], tensor[1:,:,:][neighbour_idx]])
    phasepairs = torch.cat((phasepairs,torch.transpose(neighbour_list,0,1)), 0)
    # Check y neighbours
    neighbour_idx = torch.nonzero(tensor[:,:-1,:] != tensor[:,1:,:], as_tuple=True)
    neighbour_list = torch.stack([tensor[:,:-1,:][neighbour_idx], tensor[:,1:,:][neighbour_idx]])
    phasepairs = torch.cat((phasepairs,torch.transpose(neighbour_list,0,1)), 0)
    # Check z neighbours
    neighbour_idx = torch.nonzero(tensor[:,:,:-1] != tensor[:,:,1:], as_tuple=True)
    neighbour_list = torch.stack([tensor[:,:,:-1][neighbour_idx], tensor[:,:,1:][neighbour_idx]])
    phasepairs = torch.cat((phasepairs,torch.transpose(neighbour_list,0,1)), 0)

    # Crop initial dummy values [0,0]
    phasepairs = phasepairs[1:]
    # Sort columns such that smaller number comes first
    phasepairs = torch.sort(phasepairs, dim=1)[0]
    # Remove all pairs with electrolyte/ background
    phasepairs = phasepairs[(phasepairs[:, 0] != 0) & (phasepairs[:, 1] != 0)]
    # Counts are currently not used but are essentially the pairwise surface area
    pairs, counts = torch.unique(phasepairs, dim=0, return_counts=True)
    pairs = pairs.cpu().numpy()
    label_to_index = {int(label): i for i, label in enumerate(data.grain_ids)}

    # Precompute the rotation matrices for hexagonal crystal symmetry.
    sym_ops = np.stack([
        rot.from_euler('z', 0, degrees=True).as_matrix(),
        rot.from_euler('z', 60, degrees=True).as_matrix(),
        rot.from_euler('z', 120, degrees=True).as_matrix()
    ], axis=0)  # shape: (3, 3, 3)
    
    N = pairs.shape[0]
    
    # Get Euler angles for all pairs (batch) and convert to rotation matrices
    angles0 = np.array([data.angles[label_to_index[pair[0]]] for pair in pairs])
    angles1 = np.array([data.angles[label_to_index[pair[1]]] for pair in pairs])
    R0_batch = rot.from_euler('ZXZ', angles0, degrees=True).as_matrix()  # shape (N, 3, 3)
    R1_batch = rot.from_euler('ZXZ', angles1, degrees=True).as_matrix()  # shape (N, 3, 3)
    
    # Compute the relative rotation for each pair: R_rel = R0^T @ R1.
    # Using Einstein summation to vectorize over pairs.
    R_rel = np.einsum('nij,njk->nik', np.transpose(R0_batch, (0,2,1)), R1_batch)  # shape (N, 3, 3)
    
    # Initialize an array to store maximum dot products for each pair (for each axis)
    max_dps = np.zeros((N, 3))
    
    # Loop over the symmetry operators on left and right.
    for sym1 in sym_ops:
        Q_left = sym1 @ R_rel
        for sym2 in sym_ops:
            Q = Q_left @ sym2
            # Dot products are the absolute of diagonal elements for each pair
            dps = np.abs(np.diagonal(Q, axis1=1, axis2=2))  # shape (N,3)
            max_dps = np.maximum(max_dps, dps)

    return pairs, np.degrees(np.arccos(max_dps))


def plot_global_grain_alignment(data: VoxelizedMeatball):
    """
    Plots the dot product values versus grain IDs.
    """
    dp_values = evaluate_global_grain_alignment(data)
    
    plt.figure(figsize=(8, 5))
    plt.plot(data.grain_ids, dp_values, 'o', color='blue')
    plt.xlabel("Grain ID")
    plt.ylabel("Dot Product")
    plt.title("Grain alignment given as dot product between radial direction and grain c-axis")
    plt.grid(True)
    plt.show()


def plot_relative_grain_alignment(data: VoxelizedMeatball):
    """
    Plots the dot product (misalignment) of neighbouring grains.
    """
    _, disorientations = evaluate_relative_grain_alignment(data)
    
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(disorientations[:,0])), disorientations[:,0], 'o', color='blue', label='a-axes')
    plt.plot(np.arange(len(disorientations[:,2])), disorientations[:,2], '*', color='red', label='c-axes')
    # plt.plot(np.arange(len(misorientations[:,1])), misorientations[:,1], 'x', color='black')
    plt.xlabel("Index of grain pair")
    plt.ylabel("Angle")
    plt.legend(loc="upper right")
    plt.title("Grain disorientation angle between neighbouring grains")
    plt.grid(True)
    plt.show()