"""
https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/mesh.py
"""

from skimage.measure import marching_cubes
import pyvista as pv
import os
import torch
import numpy as np
import pymskt as mskt

from GenerativeAnatomy.sdf.datasets import norm

def scale_mesh(new_mesh, old_mesh, scale_method='max_rad'):
    if type(old_mesh) is str:
        old_mesh = mskt.mesh.io.read_vtk(old_mesh)
    old_pts = mskt.mesh.get_mesh_physical_point_coords(old_mesh)
    new_pts = mskt.mesh.get_mesh_physical_point_coords(new_mesh)

    mean_old = np.mean(old_pts, axis=0)
    old_pts -= mean_old

    if scale_method == 'max_rad':
        scale = np.max(norm(old_pts), axis=-1)
        new_pts *= scale
    
    else:
        raise NotImplementedError
    new_pts += mean_old

    mskt.mesh.meshTools.set_mesh_physical_point_coords(new_mesh, new_pts)

def create_mesh(
    decoder,
    latent_vector,
    n_pts_per_axis=256,
    voxel_origin=(-1, -1, -1),
    voxel_size=None,
    batch_size=32**3,
    scale=None,
    offset=None,
    path_save=None,
    filename='mesh.vtk',
    path_original_mesh=None,
    scale_to_original_mesh=True
):

    if voxel_size is None:
        voxel_size = 2.0 / (n_pts_per_axis - 1)

    decoder.eval()

    samples = create_grid_samples(n_pts_per_axis, voxel_origin, voxel_size)
    sdf_values = get_sdfs(decoder, samples, latent_vector, batch_size)

    # resample SDFs into a grid: 
    sdf_values = sdf_values.reshape(n_pts_per_axis, n_pts_per_axis, n_pts_per_axis)

    # create mesh from gridded SDFs
    mesh = sdf_grid_to_mesh(sdf_values, voxel_origin, voxel_size, scale, offset)

    if (path_original_mesh is not None) and scale_to_original_mesh:
        scale_mesh(mesh, path_original_mesh)

    if path_save is not None:
        mesh.save(os.path.join(path_save, filename))
    return mesh

def sdf_grid_to_mesh(
    sdf_values,
    voxel_origin,
    voxel_size,
    scale=None,
    offset=None,
):
    sdf_values = sdf_values.cpu().numpy()

    verts, faces, normals, values = marching_cubes(
        sdf_values, 
        level=0, 
        spacing=(voxel_size, voxel_size, voxel_size)
    )

    verts += voxel_origin

    # additional scale / offset: 
    # this could be stored info from normalizing? 
    # e.g., the center/normalizing to be in r = 1 unit sphere
    if scale is not None:
        verts /= scale
    if offset is not None:
        verts -= offset
    
    faces_ = []
    for face_idx in range(faces.shape[0]):
        face = np.insert(faces[face_idx, :], 0, faces.shape[1])
        faces_.append(face)
    
    faces = np.hstack(faces_)
    mesh = pv.PolyData(verts, faces)

    return mesh


def create_grid_samples(
    n_pts_per_axis=256,
    voxel_origin=(-1, -1, -1),
    voxel_size=None,
):    
    n_pts_total = n_pts_per_axis ** 3

    indices = torch.arange(0, n_pts_total, out=torch.LongTensor())
    samples = torch.zeros(n_pts_total, 3)
    
    # generate samples on a grid... 
    samples[:, 2] = indices % n_pts_per_axis
    samples[:, 1] = (indices // n_pts_per_axis) % n_pts_per_axis
    samples[:, 0] = ((indices // n_pts_per_axis) // n_pts_per_axis) % n_pts_per_axis

    # scale & transform the grid as appropriate
    samples[:, :3] = samples[:, :3] * voxel_size
    for axis in range(3): 
        samples[:, axis] = samples[:, axis] + voxel_origin[axis]

    return samples

def get_sdfs(decoder, samples, latent_vector, batch_size=32**3):

    n_pts_total = samples.shape[0]

    current_idx = 0
    sdf_values = torch.zeros(samples.shape[0])

    while current_idx < n_pts_total:
        current_batch_size = min(batch_size, n_pts_total - current_idx)
        sampled_pts = samples[current_idx : current_idx + current_batch_size, :3].cuda()
        sdf_values[current_idx : current_idx + current_batch_size] = decode_sdf(
            decoder, latent_vector, sampled_pts
        ).squeeze(1).detach().cpu()

        current_idx += current_batch_size
    
    return sdf_values
    


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], dim=1)
    
    return decoder(inputs)

