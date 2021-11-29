import json
import os
import sys
from math import atan2, radians

import numpy as np
import bpy
from mathutils import Matrix, Vector, Quaternion


ROOT_DIR = bpy.path.abspath('//')
DATA_BASE = os.path.join(ROOT_DIR, 'data')
OUTPUT_BASE = os.path.join(ROOT_DIR, 'output')


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


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


def render_spherical(radius, world_up_axis, stare_center, view_count, 
                     frame_start_index=0,
                     output_base='nerf',
                     split='train'):
    os.makedirs(os.path.join(output_base, split), exist_ok=True)
    frame_index = frame_start_index
    
    # Data to store in JSON file
    out_data = {
        'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    }
    out_data['frames'] = []
    
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
        output_filename = 'r_{0:05d}.png'.format(frame_index)
        scene.render.filepath = os.path.join(output_base, split, output_filename)
        bpy.ops.render.render(write_still=True)
        frame_data = {
            'file_path': './' + split + '/r_{0:05d}'.format(frame_index),
            'rotation': horizontal_step / 2.0,
            'transform_matrix': listify_matrix(camera.matrix_world)
        }
        out_data['frames'].append(frame_data)

        phi += vertical_step
        theta += horizontal_step
        frame_index += 1

    with open(os.path.join(output_base, 'transforms_{}.json'.format(split)), 'w') as out_file:
        json.dump(out_data, out_file, indent=4)
        
    return frame_index


if __name__ == '__main__':
    scene_name = 'cube'
    resolution = (1024, 768)
    radius = 2.5
    world_up_axis = 'z'
    output_base = os.path.join(OUTPUT_BASE, 'nerf', scene_name)
    
    # Render settings
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    # Camera intrinsics setting
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    scene = bpy.context.scene
    camera = scene.camera
    camera.data.angle = 2 * atan2(*resolution)  # Vertical fov 90 degree, horizontal fov 106 degree

    stare_center = np.array([0, 0, 0])
    split = 'train'
    view_count = 50
    render_spherical(radius, world_up_axis, stare_center, 
                     view_count, 
                     frame_start_index=0, 
                     output_base=output_base,
                     split=split)
    split = 'test'
    view_count = 5
    render_spherical(radius, world_up_axis, stare_center, 
                     view_count, 
                     frame_start_index=0,
                     output_base=output_base,
                     split=split)
    split = 'val'
    view_count = 5
    render_spherical(radius, world_up_axis, stare_center, 
                     view_count, 
                     frame_start_index=0,
                     output_base=output_base, 
                     split=split)
