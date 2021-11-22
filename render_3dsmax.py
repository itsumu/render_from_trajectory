import os
import sys

import numpy as np
import pymxs
from pymxs import runtime as rt
from scipy.spatial.transform import Rotation

ROOT_DIR = os.path.dirname(__file__)
OUTPUT_BASE = os.path.join(ROOT_DIR, 'output')
sys.path.append(ROOT_DIR)
from generate_trajectory import *
import importlib
importlib.reload(sys.modules['generate_trajectory'])


def render_poses(camera, poses, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(poses)):
        position = poses[i][:3, -1]
        rotation = poses[i][:3, :3]
        if hasattr(camera, 'rotation'):
            camera.rotation = rt.Matrix3(rt.Point3(float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2])),
                                         rt.Point3(float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2])),
                                         rt.Point3(float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2])),
                                         rt.Point3(0, 0, 0))
        camera.pos = rt.Point3(float(position[0]), float(position[1]), float(position[2]))
        output_path = os.path.join(output_dir, 'image_{:05d}.jpg'.format(i))
        rt.render(camera=camera, outputFile=output_path, vfb=False)

    
def save_poses(poses, output_dir, centrialize=False, scale=False,
                pos_offset=np.array([0, 0, 0]), scale_factor=1.0):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(poses)):
        pose = poses[i]
        if centrialize:
            pose[:3, -1] = pose[:3, -1] + pos_offset
        if scale:
            pose[:3, -1] = pose[:3, -1] * scale_factor
        np.savetxt(os.path.join(output_dir, 'pose_{:05d}.txt'.format(i)), pose)
    
    
def render_images(scene_name, mode, do_render, do_save_pose,
                  scale_position, scale_factor):
    # Global objects
    main_camera = rt.getActiveCamera()
    
    # Hemisphere configs
    view_count = 100
    world_up = np.array([0, 0, 1])
    stare_center = None
    if main_camera.target is not None:
        stare_center = np.array(main_camera.target.pos)
    origin = np.array(rt.getNodeByName('Origin').pos)
    radius = 30000
    
    # Grid configs
    base_height = 10000
    side_lengths = np.array([40000, 40000, 20000])
    interval = 2500
    grid_size = (side_lengths / interval + 1).astype(np.int)
    grid_origin = np.array([origin[0] - side_lengths[0] / 2,
                            origin[1] - side_lengths[1] / 2,
                            base_height])
    default_forward = np.array([0, 1, 0])
    forward = Rotation.from_euler('x', -45, degrees=True).apply(default_forward)
    
    # Raw poses generation
    poses = []
    if mode == 'hemisphere':
        poses = generate_poses_hemisphere(view_count, world_up, stare_center, radius)
    elif mode == 'mesh':
        poses = generate_poses_mesh(world_up, stare_center, os.path.join(ROOT_DIR, 'data/mesh/hemisphere/hemisphere.obj'), radius,
                                    True)
    elif mode == 'log_file':
        poses = generate_poses_log_file(origin, 1.0, os.path.join(ROOT_DIR, 'data/trajectories/nest/traj_zhang.log'))
    elif mode == 'grid_box':
        poses = generate_poses_grid_box(grid_origin, interval, grid_size, world_up,
                                        forward, disturb=True)
    else:
        print('Render mode not exist!')
        exit(1)
        
    # Render / Save poses
    camera_default_pos = np.array(main_camera.pos)
    camera_default_rot = None
    if hasattr(main_camera, 'rotation'):
        camera_default_rot = main_camera.rotation
    pos_offset = None
    if stare_center is not None:
        pos_offset = -stare_center
    else:
        pos_offset = -origin
    
    if do_render:
        render_poses(main_camera, poses, os.path.join(OUTPUT_BASE, mode, scene_name))
        
    if do_save_pose:
        save_poses(poses, os.path.join(OUTPUT_BASE, mode, scene_name),
                   True, scale_position, pos_offset, scale_factor)
        
    main_camera.pos = camera_default_pos
    if hasattr(main_camera, 'rotation'):
        main_camera.rotation = camera_default_rot
            

if __name__ == '__main__':
    scene_name = 'nest'
    mode = 'grid_box'
    do_render = False
    do_save_pose = True
    scale_position = True
    scale_factor = 0.0001
    
    render_images(scene_name, mode, do_render, do_save_pose,
                  scale_position, scale_factor)
    print('Done')
    