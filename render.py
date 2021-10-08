import os
import sys
from math import atan2, radians
import configargparse

import numpy as np
import bpy
from mathutils import Matrix, Vector, Quaternion, Euler


ROOT_DIR = bpy.path.abspath('//')
DATA_BASE = os.path.join(ROOT_DIR, 'data')
OUTPUT_BASE = os.path.join(ROOT_DIR, 'output')


def parse_args(argv):
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument('--scene', type=str, default='default', help='Scene name')
    parser.add_argument('--output_dir', type=str, default='default', help='Output name (Same as scene most time)')
    parser.add_argument('--world_up_axis', type=str, default='y', help='World up axis')
    parser.add_argument('--resolution_x', type=int, default=800, help='Image resolution along x axis')
    parser.add_argument('--resolution_y', type=int, default=600, help='Image resolution along y axis')
    parser.add_argument('--fov_x', type=float, default=60.0, help='Horizontal fov in degrees')
    parser.add_argument('--x_min', type=float, default=-22, help='x min')
    parser.add_argument('--x_max', type=float, default=22, help='x max')
    parser.add_argument('--y_min', type=float, default=-6, help='y min')
    parser.add_argument('--y_max', type=float, default=6, help='y max')
    parser.add_argument('--z_min', type=float, default=0, help='z min')
    parser.add_argument('--z_max', type=float, default=0, help='z max')
    parser.add_argument('--const_depth', type=float, default=-1, help='Manual depth')
    parser.add_argument('--view_count_x', type=int, default=5, help='View count along x axis')
    parser.add_argument('--view_count_y', type=int, default=3, help='View count along y axis')
    parser.add_argument('--view_count_z', type=int, default=10, help='View count along z axis')
    parser.add_argument('--view_count_all', type=int, default=0, help='Overall view count')
    parser.add_argument('--sparse_view_count_x', type=int, default=9, help='Sparse view count along x axis')
    parser.add_argument('--dense_view_count_x', type=int, default=9, help='Dense view count along x axis')
    parser.add_argument('--spiral_view_count', type=int, default=9, help='Spiral view count')
    parser.add_argument('--radius', type=float, default=4.0, help='Radius for spherical render')
    parser.add_argument('--mode', type=str, default='sparse', help='Sampling mode', required=True)
    parser.add_argument('--rotate_axis', type=str, default='y')
    parser.add_argument('--position', type=float, action='append', help='Static position')
    
    return parser.parse_args(argv)


# Assume camera up is same as world up (positive y)
def look_at(camera, target_position, world_up_axis='y'):
    if world_up_axis == 'y':
        world_up = Vector((0, 1, 0))
    elif world_up_axis == 'z':
        world_up = Vector((0, 0, 1))
    forward = target_position - camera.matrix_world.to_translation()
    right = forward.cross(world_up)
    up = right.cross(forward)
    forward.normalize()
    right.normalize()
    up.normalize()
    rotation_mat = Matrix((right, up, -forward)).transposed().to_4x4()
    print(rotation_mat)
    camera.matrix_world = camera.matrix_world @ rotation_mat


def render_spiral(radii, world_up_axis, center_position, stare_center, view_count,
                  frame_start_index=0, output_dir=os.path.join(OUTPUT_BASE, 'spiral'),
                  move_back=False):
    os.makedirs(output_dir, exist_ok=True)
    frame_index = frame_start_index

    increment = 0
    if move_back: # Last frame same as the first one
        if view_count > 1:
            increment = 1 / (view_count - 1)
    else: # Last frame not same as the first one
        if view_count > 0:
            increment = 1 / view_count

    # Render
    for i in range(view_count):
        x_coords = -np.cos(2 * np.pi * increment * i) * radii[0] + center_position[0]
        y_coords = np.sin(2 * np.pi * increment * i) * radii[1] + center_position[1]
        z_coords = np.sin(2 * np.pi * increment * i) * radii[2] + center_position[2]
        current_position = np.array([x_coords, y_coords, z_coords])
        camera.matrix_world = Matrix.Translation(current_position)
        look_at(camera, Vector(stare_center), world_up_axis)
        output_filename = 'image_{:03d}.jpg'.format(frame_index)
        scene.render.filepath = os.path.join(output_dir, output_filename)
        bpy.ops.render.render(write_still=True)
        np.savetxt(os.path.join(output_dir, 'pose_{:03d}.txt'.format(frame_index)), camera.matrix_world)

        frame_index += 1

    return frame_index


def render_forward_grid(start_position, x_interval, y_interval, view_count_x, view_count_y, frame_start_index=0, output_dir=os.path.join(OUTPUT_BASE, 'forward_grid')):
    os.makedirs(output_dir, exist_ok=True)
    frame_index = frame_start_index

    # Render
    for j in range(view_count_y):
        for i in range(view_count_x):
            current_position = start_position + np.array([x_interval, 0, 0]) * i + np.array([0,
             y_interval, 0]) * j
            camera.matrix_world = Matrix.Translation(current_position)
            output_filename = 'image_{:03d}.jpg'.format(frame_index)
            scene.render.filepath = os.path.join(output_dir, output_filename)
            scene.frame_set(frame_index)
            bpy.ops.render.render(write_still=True)
            np.savetxt(os.path.join(output_dir, 'pose_{:03d}.txt'.format(frame_index)), camera.matrix_world)

            frame_index += 1

    return frame_index


def render_grid_center(start_position, world_up_axis, stare_center, 
                       x_interval, y_interval, view_count_x, view_count_y, 
                       frame_start_index=0, output_dir=os.path.join(OUTPUT_BASE, 'grid_center')):
    os.makedirs(output_dir, exist_ok=True)
    frame_index = frame_start_index

    # Render
    for j in range(view_count_y):
        for i in range(view_count_x):
            current_position = start_position + np.array([x_interval, 0, 0]) * i \
                               + np.array([0, y_interval, 0]) * j
            camera.matrix_world = Matrix.Translation(current_position)
            look_at(camera, Vector(stare_center), world_up_axis)
            output_filename = 'image_{:03d}.jpg'.format(frame_index)
            scene.render.filepath = os.path.join(output_dir, output_filename)
            bpy.ops.render.render(write_still=True)
            np.savetxt(os.path.join(output_dir, 'pose_{:03d}.txt'.format(frame_index)), camera.matrix_world)

            frame_index += 1

    return frame_index


def render_stare_center_x_axis(start_position, x_interval, view_count_x, frame_start_index=0, output_dir=os.path.join(OUTPUT_BASE, 'stare_center_x_axis')):
    os.makedirs(output_dir, exist_ok=True)
    frame_index = frame_start_index

    # Render
    for i in range(view_count_x):
        current_position = start_position + np.array([x_interval, 0, 0]) * i
        camera.matrix_world = Matrix.Translation(current_position)
        look_at(camera, Vector((0, 0, 0)))
        output_filename = 'image_{:03d}.jpg'.format(frame_index)
        scene.render.filepath = os.path.join(output_dir, output_filename)
        scene.frame_set(frame_index)
        bpy.ops.render.render(write_still=True)
        np.savetxt(os.path.join(output_dir, 'pose_{:03d}.txt'.format(frame_index)), camera.matrix_world)

        frame_index += 1

    return frame_index


def render_zoom_in(start_position, z_interval, view_count_z, frame_start_index=0, output_dir=os.path.join(OUTPUT_BASE, 'zoom_in')):
    os.makedirs(output_dir, exist_ok=True)
    frame_index = frame_start_index

    for i in range(view_count_z):
        current_position = start_position + np.array([0, 0, z_interval]) * i
        camera.matrix_world = Matrix.Translation(current_position)
        output_filename = 'image_{:03d}.jpg'.format(i)
        scene.render.filepath = os.path.join(output_dir, output_filename)
        scene.frame_set(frame_index)
        bpy.ops.render.render(write_still=True)
        np.savetxt(os.path.join(output_dir, 'pose_' + str(frame_index) + '.txt'), camera.matrix_world)

        frame_index += 1

    return frame_index


def render_spherical(radius, world_up_axis, stare_center, view_count, 
                     frame_start_index=0, 
                     output_dir=os.path.join(OUTPUT_BASE, 'spherical')):
    os.makedirs(output_dir, exist_ok=True)
    frame_index = frame_start_index

    vertical_step = radians(180.0 / view_count)
    horizontal_step = radians(360.0 / (view_count / 2.0))
    phi = 0.0
    theta = 0.0
    # Render
    for i in range(view_count):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = radius * np.cos(phi)
        camera.matrix_world = Matrix.Translation((x, y, z))
        look_at(camera, Vector(stare_center), world_up_axis)
        output_filename = 'image_{:03d}.jpg'.format(frame_index)
        scene.render.filepath = os.path.join(output_dir, output_filename)
        bpy.ops.render.render(write_still=True)
        np.savetxt(os.path.join(output_dir,
                                'pose_{:03d}.txt'.format(frame_index)),
                   camera.matrix_world)

        phi += vertical_step
        theta += horizontal_step
        frame_index += 1
        
    return frame_index


def render_hemisphere(radius, world_up_axis, stare_center, view_count, 
                      frame_start_index=0, 
                      output_dir=os.path.join(OUTPUT_BASE, 'hemisphere')):
    os.makedirs(output_dir, exist_ok=True)
    frame_index = frame_start_index
    
    # Sample positions
    positions = generate_hemisphere_data(stare_center, radius, view_count)
    
    # Render
    for i in range(view_count):
        if world_up_axis == 'z':
            x, y, z = positions[i]
        elif world_up_axis == 'y':
            z, x, y = positions[i]
        else:
            return
        camera.matrix_world = Matrix.Translation((x, y, z))
        look_at(camera, Vector(stare_center), world_up_axis)
        output_filename = 'image_{:03d}.jpg'.format(frame_index)
        scene.render.filepath = os.path.join(output_dir, output_filename)
        bpy.ops.render.render(write_still=True)
        np.savetxt(os.path.join(output_dir,
                                'pose_{:03d}.txt'.format(frame_index)),
                   camera.matrix_world)

        frame_index += 1
        
    return frame_index


def render_rotate(position, world_up_axis, rotate_axis, view_count,
                  frame_start_index, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_index = frame_start_index

    theta = radians(360.0 / view_count)
    if world_up_axis == 'z':
        camera.matrix_world = Matrix.Rotation(radians(90), 4, 'X') @ camera.matrix_world 
    camera.matrix_world = Matrix.Translation(position) @ camera.matrix_world
    # Render
    for i in range(view_count):
        camera.matrix_world = Matrix.Translation(position) \
            @ Matrix.Rotation(theta, 4, rotate_axis.upper()) \
            @ Matrix.Translation(-position) \
            @ camera.matrix_world
        output_filename = 'image_{:03d}.jpg'.format(frame_index)
        scene.render.filepath = os.path.join(output_dir, output_filename)
        bpy.ops.render.render(write_still=True)
        np.savetxt(os.path.join(output_dir,
                                'pose_{:03d}.txt'.format(frame_index)),
                   camera.matrix_world)

        frame_index += 1
        
    return frame_index


def generate_training_data(x_range, y_range, x_interval, y_interval, view_count_x, view_count_y,
 depth, scene_name):
    frame_index = 0
    train_dir = os.path.join(OUTPUT_BASE, 'train', scene_name)

    # Render a xy-plane grid looking towards -z
    start_position = np.array([x_range[0], y_range[0], depth])
    render_forward_grid(start_position, x_interval, y_interval, view_count_x, view_count_y, frame_index, train_dir)


def generate_validation_data(x_range, y_range, x_interval, y_interval, view_count_x, view_count_y,
 depth, scene_name):
    frame_index = 0
    val_dir = os.path.join(OUTPUT_BASE, 'val', scene_name)

    # Render the in-between views of training views
    start_position = np.array([x_range[0] + x_interval / 2, 0, depth])
    frame_index = render_forward_grid(start_position, x_interval, y_interval, view_count_x - 1, 1, frame_index, val_dir)

    # From left to right looking at origin
    start_position = np.array([x_range[0], 0, depth * 2])

    frame_index = render_stare_center_x_axis(start_position, x_interval, view_count_x, frame_index, val_dir)

    # Render the center view of source ones
    start_position = np.array([x_range[0], 0, depth])
    render_forward_grid(start_position, x_interval, 0, view_count_x, 1, frame_index, val_dir)


def generate_sparse_data(x_range, x_interval, view_count_x, depth, scene_name):
    frame_index = 0
    sparse_dir = os.path.join(OUTPUT_BASE, 'sparse', scene_name)

    start_position = np.array([x_range[0], 0, depth])

    render_stare_center_x_axis(start_position, x_interval, view_count_x, frame_index, sparse_dir)


def generate_dense_data(x_range, x_interval, view_count_x, depth, scene_name):
    frame_index = 0
    sparse_dir = os.path.join(OUTPUT_BASE, 'dense', scene_name)

    start_position = np.array([x_range[0], 0, depth])

    render_stare_center_x_axis(start_position, x_interval, view_count_x, frame_index, sparse_dir)


def generate_linear_data(start_position, end_position, view_count, scene_name):
    output_dir = os.path.join(OUTPUT_BASE, 'linear', scene_name)
    os.makedirs(output_dir, exist_ok=True)
    frame_index = 0
    interval = (end_position - start_position) / (view_count - 1)

    for i in range(view_count):
        current_position = start_position + interval * i
        camera.matrix_world = Matrix.Translation(current_position)
        output_filename = 'image_{:03d}.jpg'.format(frame_index)
        scene.render.filepath = os.path.join(output_dir, output_filename)
        scene.frame_set(frame_index)
        bpy.ops.render.render(write_still=True)
        np.savetxt(os.path.join(output_dir, 'pose_{:03d}.txt'.format(frame_index)), camera.matrix_world)

        frame_index += 1


def generate_manual_data(trajectories_dir, output_dir):
    for frame_index, file_name in enumerate(sorted(os.listdir(trajectories_dir))):
        # Load from file
        file_path = os.path.join(trajectories_dir, file_name)
        pose = np.loadtxt(file_path)

        # Move camera
        camera.matrix_world = Matrix(pose)
        output_filename = 'image_{:03d}.jpg'.format(frame_index)
        scene.render.filepath = os.path.join(output_dir, output_filename)
        scene.frame_set(frame_index)
        bpy.ops.render.render(write_still=True)
        np.savetxt(os.path.join(output_dir, 'pose_{:03d}.txt'.format(frame_index)), camera.matrix_world)


def generate_hemisphere_data(center, radius, sample_count):
    # Best-candidate algorithm
    candidate_count = 10
    samples = []
    for i in range(sample_count): # For each required sample
        best_candidate_dist = 0
        best_candidate = np.array([10, 10, 10])
        for j in range(candidate_count): # For each candidate, find distant one
            new_candidate_xy = np.random.uniform(-1, 1, 2)
            while (np.linalg.norm(new_candidate_xy) > 1):
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
        samples[i] = samples[i] * radius + center
    return samples

def setup_renderer(scene):
    # Renderer & device settings
    scene.render.engine = 'CYCLES'
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type =\
        'CUDA'
    scene.cycles.device = 'GPU'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # Data settings
    scene.render.use_persistent_data = True
    
    # Output settings
    scene.render.image_settings.color_mode = 'RGBA'
    

if __name__ == '__main__':
    # Parse arguments
    argv = sys.argv
    if "--" not in argv:
        argv = []  # No python args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    args = parse_args(argv)

    # Render settings
    scene = bpy.context.scene
    setup_renderer(scene)
    
    # Camera intrinsics setting
    resolution = (args.resolution_x, args.resolution_y)
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    camera = scene.camera
    camera.data.angle_x = args.fov_x / 180.0 * np.pi

    # Camera movement boundaries, count and corresponding intervals
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)
    view_count_x = args.view_count_x
    view_count_y = args.view_count_y

    if view_count_x != 1:
        x_interval = (x_range[1] - x_range[0]) / (view_count_x - 1)
    else:
        x_interval = 0
    if view_count_y != 1:
        y_interval = (y_range[1] - y_range[0]) / (view_count_y - 1)
    else:
        y_interval = 0
    if args.const_depth == -1:
        depth = y_interval # Overlap 50% for fov 90 degree
    else:
        depth = args.const_depth

    z_range = (args.z_min, args.z_max)
    view_count_z = args.view_count_z
    if view_count_z != 1:
        z_interval = (z_range[1] - z_range[0]) / (view_count_z - 1)
    else:
        z_interval = 0

    if args.output_dir == 'default': args.output_dir = args.scene
    if args.mode == 'train':
        generate_training_data(x_range, y_range, x_interval, y_interval, view_count_x, view_count_y,
         depth, args.scene)
    elif args.mode == 'val':
        generate_validation_data(x_range, y_range, x_interval, y_interval, view_count_x, view_count_y,
         depth, args.scene)
    elif args.mode == 'zoom':
        frame_index = 0
        zoom_dir = os.path.join(OUTPUT_BASE, 'zoom', args.scene)

        z_range = (depth - depth / 2, depth + depth / 2)
        z_interval = (z_range[1] - z_range[0]) / (view_count_z - 1)
        start_position = np.array([0, 0, z_range[0]])
        render_zoom_in(start_position, z_interval, view_count_z, frame_index, zoom_dir)
    elif args.mode == 'sparse':
        depth = args.const_depth
        view_count_x = args.sparse_view_count_x
        x_interval = (x_range[1] - x_range[0]) / (view_count_x - 1)
        generate_sparse_data(x_range, x_interval, view_count_x, depth, args.scene)
    elif args.mode == 'dense':
        depth = args.const_depth
        view_count_x = args.dense_view_count_x
        x_interval = (x_range[1] - x_range[0]) / (view_count_x - 1)
        generate_dense_data(x_range, x_interval, view_count_x, depth, args.scene)
    elif args.mode == 'linear':
        start_position = np.array([x_range[0], y_range[0], z_range[0]])
        end_position = np.array([x_range[1], y_range[1], z_range[1]])
        generate_linear_data(start_position, end_position, args.view_count_all, args.scene)
    elif args.mode == 'forward':
        start_position = np.array([x_range[0], y_range[0], args.const_depth])
        render_forward_grid(start_position, x_interval, y_interval, view_count_x, view_count_y, output_dir=os.path.join(OUTPUT_BASE, 'forward', args.output_dir))
    elif args.mode == 'spiral':
        start_position = np.array([x_range[0], y_range[0], z_range[0]])
        x_radius = (x_range[1] - x_range[0]) / 2
        y_radius = (y_range[1] - y_range[0]) / 2
        z_radius = (z_range[1] - z_range[0]) / 2
        radii = np.array([x_radius, y_radius, z_radius])
        center_position = np.array([(x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, (z_range[0] + z_range[1]) / 2])
        stare_center = np.array([0, 0, 0])
        render_spiral(radii, args.world_up_axis, center_position, stare_center, args.spiral_view_count, output_dir=os.path.join(OUTPUT_BASE, 'spiral', args.output_dir))
    elif args.mode == 'manual':
        generate_manual_data(os.path.join(DATA_BASE, 'trajectories', args.scene), os.path.join(OUTPUT_BASE, 'manual', args.output_dir))
    elif args.mode == 'grid_center':
        start_position = np.array([x_range[0], y_range[0], z_range[0]])
        stare_center = np.array([0, 0, 0])
        render_grid_center(start_position, args.world_up_axis, stare_center, 
            x_interval, y_interval, view_count_x, view_count_y,
            output_dir=os.path.join(OUTPUT_BASE, 'grid_center', args.output_dir))
    elif args.mode == 'spherical':
        stare_center = np.array([0, 0, 0])
        render_spherical(args.radius, args.world_up_axis, stare_center, 
                         args.view_count_all, 
                         frame_start_index=0, 
                         output_dir=os.path.join(OUTPUT_BASE, 'spherical',
                                                 args.output_dir))
    elif args.mode == 'hemisphere':
        stare_center = np.array([0, 0, 0])
        render_hemisphere(args.radius, args.world_up_axis, stare_center,
                          args.view_count_all, 
                          frame_start_index=0, 
                          output_dir=os.path.join(OUTPUT_BASE, 'hemisphere',
                                                  args.output_dir))
    elif args.mode == 'rotate':
        render_rotate(np.array(args.position),
                      args.world_up_axis,
                      args.rotate_axis,
                      args.view_count_all,
                      frame_start_index=0, 
                      output_dir=os.path.join(OUTPUT_BASE, 'rotate',
                                              args.output_dir))
    else:
        print('Render mode not specified!')