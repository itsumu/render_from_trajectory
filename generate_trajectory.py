import math

import numpy as np


def look_at(camera_pos, world_up, target_pos, camera_up): # Camera up not used for now
    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)
    while forward @ world_up == 1 or forward @ world_up == -1:
        world_up = np.random.rand(3)
        world_up = world_up / np.linalg.norm(world_up)
    right = np.cross(forward, world_up)
    up = np.cross(right, forward)
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    rotation_mat = np.stack((right, up, -forward), axis=-1) # Stack as columns
    pose_mat = np.concatenate((rotation_mat, np.expand_dims(camera_pos, -1)),
                              -1)
    pose_mat = np.concatenate((pose_mat, np.array([[0, 0, 0, 1]])))
    return pose_mat
    
def sample_positions_on_sphere(origin, radius, count_on_perimeter):
    assert count_on_perimeter > 4
    theta_range = math.pi
    theta_interval = 2 * math.pi / count_on_perimeter
    count_on_quadrant = math.floor(theta_range / theta_interval) + 1
    
    phi = 0
    positions = []
    for i in range(count_on_quadrant):
        theta = i * theta_interval
        longitude_count = max(math.floor(count_on_perimeter * math.sin(theta)), 1)
        phi_interval = 2 * math.pi / longitude_count
        for j in range(longitude_count):
            phi += phi_interval
            position = np.array([math.sin(theta) * math.cos(phi),
                                 math.sin(theta) * math.sin(phi),
                                 math.cos(theta)])
            position = origin + radius * position
            positions.append(position)
    return positions
    
def sample_positions_on_hemisphere(sample_count, origin, radius, mode='uniform'):
    samples = []
    if mode == 'random':
        for i in range(sample_count):
            xy_coords = np.random.uniform(-1, 1, 2)
            while np.linalg.norm(xy_coords) > 1:
                xy_coords = np.random.uniform(-1, 1, 2)        
            z_coord = np.sqrt(1 - xy_coords[0] * xy_coords[0] -
                            xy_coords[1] * xy_coords[1])
            samples.append(np.array([xy_coords[0], xy_coords[1], z_coord]))
    elif mode == 'uniform':
        # Best-candidate algorithm
        candidate_count = 10
        for i in range(sample_count): # For each required sample
            best_candidate_dist = 0
            best_candidate = np.array([10, 10, 10])
            for j in range(candidate_count): # For each candidate, find distant one
                new_candidate_xy = np.random.uniform(-1, 1, 2)
                while np.linalg.norm(new_candidate_xy) > 1:
                    new_candidate_xy = np.random.uniform(-1, 1, 2)
                new_candidate_z = np.sqrt(1 - 
                                        new_candidate_xy[0] * new_candidate_xy[0] -
                                        new_candidate_xy[1] * new_candidate_xy[1])
                new_candidate = np.array([new_candidate_xy[0],
                                        new_candidate_xy[1],
                                        new_candidate_z])
                if i == 0:
                    best_candidate = new_candidate
                    break                
                
                smallest_dist = 100
                for sample in samples: # For each fixed sample, find smallest distance
                    dist = np.arccos(sample @ new_candidate)
                    smallest_dist = min(smallest_dist, dist)
                if smallest_dist > best_candidate_dist:
                    best_candidate = new_candidate
                    best_candidate_dist = smallest_dist
            samples.append(best_candidate) 
            
    # All calculations before are based on origin of [0,0,0], radius of 1
    # Should be transformed to origin or "center", radius of "radius"
    for i in range(sample_count):
        samples[i] = samples[i] * radius + origin
    return samples


def generate_poses_spherical(world_up, origin, radius, count_on_perimeter):
    positions = sample_positions_on_sphere(origin, radius, count_on_perimeter)
    poses = [look_at(position, world_up, origin, world_up) for position in positions]
    
    return poses


def generate_poses_hemisphere(view_count, world_up, origin, radius):
    positions = sample_positions_on_hemisphere(view_count, origin, radius)
    poses = [look_at(position, world_up, origin, world_up) for position in positions]
    
    return poses


def generate_poses_mesh(world_up, origin, mesh_path, scale, remove_duplicate=False):
    import trimesh    

    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = mesh.vertices
    if remove_duplicate:
        dist_epsilon = 1e-3
        for i in range(len(vertices)):
            j = i + 1
            while j < len(vertices):
                if np.linalg.norm(vertices[j] - vertices[i]) < dist_epsilon:
                    vertices = np.delete(vertices, j, axis=0)
                else:
                    j += 1
    positions = vertices * scale + origin

    poses = []
    last_cam_up = world_up
    for i in range(len(positions)):
        new_pose = look_at(positions[i], world_up, origin, last_cam_up)
        poses.append(new_pose)
        last_cam_up = new_pose[:3, 1]
        
    return poses


def generate_poses_log_file(origin, scale, log_path):
    from scipy.spatial.transform import Rotation
    
    poses = []
    with open(log_path, 'r') as log_file:
        line = log_file.readline()
        while line != '':
            segments = line.split(',')
            pitch, roll, yaw = segments[4:7]
            default_mat = Rotation.from_euler('x', 90, degrees=True).as_matrix()
            pitch_mat = Rotation.from_euler('x', -float(pitch), degrees=True).as_matrix()
            yaw_mat = Rotation.from_euler('z', float(yaw), degrees=True).as_matrix()
            
            rotation_mat = yaw_mat @ pitch_mat @ default_mat
            position_vec = np.array(segments[1:4]).astype(np.float) / scale + origin
            pose = np.concatenate((rotation_mat, np.expand_dims(position_vec, axis=-1)),
                                    axis=-1)
            pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)
            
            poses.append(pose)
            line = log_file.readline()
    return poses


def generate_poses_grid_box(origin, interval, grid_size,
                            world_up, forward=np.array([0, 0, 0]), disturb=True,
                            stare_center=None, proxy_path=None):
    '''
        origin: (x_min, y_min, z_min)
        interval: (x_width, y_width, z_width) or interval
        grid_size: (x_count, y_count, z_count)
    '''
    # Optional proxy for inside/outside check
    proxy = None
    if proxy_path is not None:
        import trimesh
        proxy = trimesh.load_mesh(proxy_path)
        ray_caster = trimesh.ray.ray_triangle.RayMeshIntersector(proxy)
    # Generate poses
    poses = []
    x_count, y_count, z_count = grid_size
    for i in range(x_count):
        for j in range(y_count):
            for k in range(z_count):
                position_vec = origin + np.array([i, j, k]) * interval
                target_pos = position_vec + forward
                if stare_center is not None:
                    target_pos = stare_center
                    if np.linalg.norm(position_vec - target_pos) == 0:
                        target_pos = position_vec + np.random.uniform(size=3)
                if disturb:
                    random_disturb = np.random.rand(3)
                    target_pos += 0.1 * random_disturb / np.linalg.norm(random_disturb)
                if proxy is not None:
                    direction = target_pos - position_vec
                    if ray_caster.intersects_any(position_vec[None, :],
                                                 direction[None, :]):   
                        tri_idx = ray_caster.intersects_first(position_vec[None, :],
                                                            direction[None, :])
                        normal = proxy.face_normals[tri_idx[0]]
                        if normal @ direction > 0: # Same direction, view lies inside geometry
                            continue
                    else: # Not intersected with any geometry
                        continue
                pose = look_at(position_vec, world_up, target_pos, world_up)
                poses.append(pose)
    return poses
