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


def generate_poses_hemisphere(view_count, world_up, origin, radius):
    positions = sample_positions_on_hemisphere(view_count, origin, radius)
    poses = [look_at(position, world_up, origin) for position in positions]
    
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
