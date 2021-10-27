import os
import sys

import numpy as np
import pymxs
from pymxs import runtime as rt


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
        camera.pos = rt.Point3(float(position[0]), float(position[1]), float(position[2]))
        output_path = os.path.join(output_dir, 'image_{:03d}.jpg'.format(i))
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
        np.savetxt(os.path.join(output_dir, 'pose_{:03d}.txt'.format(i)), pose)
    
    
def render_images():
    # Global objects
    main_camera = rt.getActiveCamera()
    
    # Hemisphere configs
    view_count = 100
    world_up = np.array([0, 0, 1])
    stare_center = np.array(main_camera.target.pos)
    radius = 30000
    
    # Global configs
    scene_name = 'nest'
    mode = 'mesh'
    do_render = True
    do_save_pose = True
    scale_position = True
    scale_factor = 0.0001
    
    # Raw poses generation
    poses = []
    if mode == 'hemisphere':
        poses = generate_poses_hemisphere(view_count, world_up, stare_center, radius)
    elif mode == 'mesh':
        poses = generate_poses_mesh(world_up, stare_center, os.path.join(ROOT_DIR, 'data/mesh/hemisphere/hemisphere.obj'), radius,
                                    True)
    else:
        print('Render mode not exist!')
        exit(1)
        
    # Render / Save poses
    camera_default_pos = np.array(main_camera.pos)
    pos_offset = -stare_center
    
    if do_render:
        render_poses(main_camera, poses, os.path.join(OUTPUT_BASE, mode, scene_name))
        
    if do_save_pose:
        save_poses(poses, os.path.join(OUTPUT_BASE, mode, scene_name),
                   True, scale_position, pos_offset, scale_factor)
        
    main_camera.pos = camera_default_pos
        
if __name__ == '__main__':
    render_images()
    print('Done')
    