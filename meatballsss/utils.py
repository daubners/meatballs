import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as rot

def generate_random_points_within_sphere(num_points, radius):
    points = np.random.randn(num_points, 3)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    points *= (np.random.rand(num_points, 1) ** (1/3)) * radius
    return points

def generate_fibonacci_points_on_sphere(num_points, radius):
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    indices = np.arange(num_points)
    y = 1 - (indices / float(num_points - 1)) * 2  # y goes from 1 to -1
    rad = np.sqrt(1 - y * y)  # radius at y
    theta = phi * indices  # golden angle increment

    x = np.cos(theta) * rad * radius
    z = np.sin(theta) * rad * radius

    points = np.vstack((x, y * radius, z)).T
    return points

def normalize_angles(raw_angles):
    norm_angles = np.copy(raw_angles)
    change_signs = norm_angles[:,0]<0
    norm_angles[change_signs, 0] += 180
    norm_angles[change_signs, 1] *= -1

    change_signs = norm_angles[:,1]<0
    norm_angles[change_signs, 1] += 180
    norm_angles[change_signs, 2] *= -1

    norm_angles[norm_angles[:,2]<0, 2] += 120
    norm_angles[norm_angles[:,2]<0, 2] += 120
    norm_angles[norm_angles[:,2]>=120, 2] -= 120
    
    return norm_angles

def create_NMC_agglomerate(radii, num_seeds, padding=5, return_seeds=False):
    # Radii should be sorted from smallest to largest
    # If only one radius is given, then only create random meatball
    # If N>1 radii are given, create N-1 layers of radially aligned grains and
    # randomly oriented core.
    nx = 2*(radii[-1]+padding)
    ny = nx
    nz = nx
    center = np.array([nx/2, ny/2, nz/2])

    seeds = generate_random_points_within_sphere(num_seeds[0], radii[0])

    for i in range(1,len(radii)):
        outer_seeds = generate_fibonacci_points_on_sphere(num_seeds[i], (radii[i]+radii[i-1])/2)
        seeds = np.concatenate((seeds,outer_seeds),axis=0)
    
    # elyte_seeds = generate_fibonacci_points_on_sphere(num_seeds[-1], (3*radii[-1]-radii[-2])/2)
    # seeds = np.concatenate((seeds,elyte_seeds),axis=0)

    # Add orientations and start with random list
    # angle_list = 180*np.random.rand(np.sum(num_seeds), 3)
    # Last angle is defined in [0,120]
    angle_list = np.column_stack((180*np.random.rand(np.sum(num_seeds), 2), 120*np.random.rand(np.sum(num_seeds), 1)))
    # Add orientations for radially aligned grains given in Bunge convention
    #  φ1 (phi1): Rotation angle about the Z-axis of the standard reference frame.
    #  Φ (Phi): Rotation angle about the X-axis of the intermediate frame obtained after the first rotation.
    #  φ2 (phi2): Rotation angle about the Z-axis of the final frame obtained after the first two rotations.
    # --> φ1 and Φ are given by seed position, φ2 can be random.
    if len(radii) > 1:
        yaw = np.arctan2(seeds[num_seeds[0]:,1],seeds[num_seeds[0]:,0]) * 180 / np.pi
        r = np.linalg.norm(seeds[num_seeds[0]:], axis=1)
        pitch = np.arcsin(-seeds[num_seeds[0]:,2] / r) * 180 / np.pi
        roll = 180*np.random.rand(np.sum(num_seeds[1:]), 1)
        ypr_angles = np.column_stack((yaw, pitch, roll))
        rot_obj = rot.from_euler('ZYX', ypr_angles, degrees=True)
        bunge_angles = rot_obj.as_euler('ZXZ', degrees=True)
        bunge_angles = normalize_angles(bunge_angles)
        angle_list[num_seeds[0]:,:] = bunge_angles

    # Shift seeds to box center
    seeds += center

    # Create a 3D grid of coordinates
    x, y, z = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5, np.arange(nz)+0.5, indexing='ij')
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Assign each voxel to the nearest seed and shift IDs by 1
    tree = KDTree(seeds)
    _, indices = tree.query(points, workers=-1)

    indices += 1  # Shift indices to start from 1 instead of 0 because the 0 is for the electrolyte
    grain_id = indices.reshape((nx, ny, nz))

    # crop spherical agglomerate
    # TODO: maybe there is some nicer way than just cropping a perfect sphere
    x, y, z = np.ogrid[:nx, :ny, :nz]
    distance_squared = (x - center[0] + 0.5)**2 + (y - center[1] + 0.5)**2 + (z - center[2] + 0.5)**2
    mask = distance_squared > radii[-1]**2

    # mask = grain_id > np.sum(num_seeds)
    grain_id[mask] = 0

    if return_seeds:
        return grain_id, angle_list, seeds
    else:
        return grain_id, angle_list