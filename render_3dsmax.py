import os
import sys

import configargparse
import numpy as np
import pymxs
from pymxs import runtime as rt
from scipy.spatial.transform import Rotation

ROOT_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(ROOT_DIR, 'configs')
OUTPUT_BASE = os.path.join(ROOT_DIR, 'output')
sys.path.append(ROOT_DIR)
from generate_trajectory import *
import importlib
importlib.reload(sys.modules['generate_trajectory'])


def parse_args(config_file_path):
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, default=config_file_path,
                        help='Config file path')
    parser.add_argument('--scene_name', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--base_height', type=float, help='Base camera height')
    parser.add_argument('--side_length_x', type=float, help='Grid box: Side length')
    parser.add_argument('--side_length_y', type=float, help='Grid box: Side length')
    parser.add_argument('--side_length_z', type=float, help='Grid box: Side length')
    parser.add_argument('--interval', type=float, help='Grid box: Interval')
    parser.add_argument('--extra_mesh', type=str, help='Grid box: Extra mesh for visibility check')
    parser.add_argument('--log_file', type=str, help='Log file: Log file path')
    return parser.parse_args()


def generate_poses_manual(camera_name, start_frame, end_frame, step):
    poses = []
    camera = rt.getNodeByName(camera_name)
    with pymxs.animate(True):
        for i in range(start_frame, end_frame, step):
            with pymxs.attime(i):
                pos = np.asarray(camera.pos)
                dir = -np.asarray(camera.dir)
                poses.append(look_at(pos, np.asarray([0, 0, 1]), pos + dir, [0]))
    return poses


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
        pose = np.copy(poses[i])
        if centrialize:
            pose[:3, -1] = pose[:3, -1] + pos_offset
        if scale:
            pose[:3, -1] = pose[:3, -1] * scale_factor
        np.savetxt(os.path.join(output_dir, 'pose_{:05d}.txt'.format(i)), pose)
    
    
def render_images(scene_name, mode, do_render, do_save_pose,
                  scale_position, scale_factor, args):
    # Global objects
    main_camera = rt.getActiveCamera()
    
    # Hemisphere configs
    view_count = 100
    world_up = np.array([0, 0, 1])
    stare_center = None
    if hasattr(main_camera, 'target') and main_camera.target is not None:
        stare_center = np.array(main_camera.target.pos)
    origin = np.array(rt.getNodeByName('Origin').pos)
    radius = 30000
    
    # Grid configs
    base_height = args.base_height
    side_lengths = np.array([args.side_length_x, args.side_length_y, args.side_length_z])
    interval = args.interval
    grid_size = (side_lengths / interval + 1).astype(np.int)
    grid_origin = np.array([origin[0] - side_lengths[0] / 2,
                            origin[1] - side_lengths[1] / 2,
                            base_height])
    default_forward = np.array([0, 1, 0])
    forward = Rotation.from_euler('x', -45, degrees=True).apply(default_forward)
    
    # Manual configs
    camera_name = 'CameraLine'
    start_frame = 0
    end_frame = 300
    step = 1
    
    # Raw poses generation
    poses = []
    if mode == 'hemisphere':
        poses = generate_poses_hemisphere(view_count, world_up, stare_center, radius)
    elif mode == 'mesh':
        poses = generate_poses_mesh(world_up, stare_center, os.path.join(ROOT_DIR, 'data/mesh/hemisphere/hemisphere.obj'), radius,
                                    True)
    elif mode == 'log_file':
        poses = generate_poses_log_file(origin, 1.0, os.path.join(ROOT_DIR, args.log_file))
    elif mode == 'grid_box':
        poses = generate_poses_grid_box(grid_origin, interval, grid_size, world_up,
                                        forward, disturb=True, proxy_path=args.extra_mesh)
        # poses = generate_poses_grid_box(grid_origin, interval, grid_size, world_up,
                                        # disturb=False, stare_center=origin,
                                        # proxy_path=args.extra_mesh)
    elif mode == 'manual':
        poses = generate_poses_manual(camera_name, start_frame, end_frame, step)
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
        
    if do_save_pose:
        save_poses(poses, os.path.join(OUTPUT_BASE, mode, scene_name),
                   True, scale_position, pos_offset, scale_factor)
    
    if do_render:
        render_poses(main_camera, poses, os.path.join(OUTPUT_BASE, mode, scene_name))
        
    main_camera.pos = camera_default_pos
    if hasattr(main_camera, 'rotation'):
        main_camera.rotation = camera_default_rot
            

if __name__ == '__main__':   
    args = parse_args(os.path.join(CONFIG_DIR, 'config_Olympics.txt'))
    scene_name = args.scene_name
    mode = args.mode
    do_render = False
    do_save_pose = True
    scale_position = True
    scale_factor = 0.0001
    
    render_images(scene_name, mode, do_render, do_save_pose,
                  scale_position, scale_factor, args)
    print('Done')
    