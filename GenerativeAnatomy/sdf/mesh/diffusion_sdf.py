import torch


from .main import (
    create_grid_samples,
    sdf_grid_to_mesh,
    apply_similarity_transform,
    scale_mesh,
)

def create_mesh_diffusion_sdf(
    model,
    latent_vector,
    n_pts_per_axis=256,
    voxel_origin=(-1, -1, -1),
    voxel_size=None,
    #
    batch_size=32**3,
    scale=1.0,
    offset=(0., 0., 0.),
    path_save=None,
    filename='mesh.vtk',
    path_original_mesh=None,
    scale_to_original_mesh=False,
    R=None,
    t=None,
    s=None,
    icp_transform=None,
    
    device='cuda:0',

    verbose=False,
):
    
    model.to(device)
    model.eval()
    model.vae_model.to(device)
    model.vae_model.eval()
    model.sdf_model.to(device)
    model.sdf_model.eval()

    if voxel_size is None:
        voxel_size = 2.0 / (n_pts_per_axis - 1)

    samples = create_grid_samples(n_pts_per_axis, voxel_origin, voxel_size).to(device)
    
    plane_features = model.vae_model.decode(latent_vector)
    
    sdf_values = get_sdfs_diffusion(
        model=model, 
        plane_features=plane_features, 
        samples=samples, 
        latent_vector=latent_vector,
        batch_size=batch_size
    )
    
    sdf_values = sdf_values.reshape(n_pts_per_axis, n_pts_per_axis, n_pts_per_axis)
    
    # create mesh from gridded SDFs
    if (0 < sdf_values.min()) or (0 > sdf_values.max()):
        print('WARNING: SDF values do not span 0 - there is no surface')
        if verbose is True:
            print('\tSDF min: ', sdf_values.min())
            print('\tSDF max: ', sdf_values.max())
            print('\tSDF mean: ', sdf_values.mean())
        return None
    else:
        mesh = sdf_grid_to_mesh(sdf_values, voxel_origin, voxel_size)
    
    if (R is True) & (s is True) & (t is True):
        apply_similarity_transform(mesh, R, t, s)
    elif scale_to_original_mesh:
        if verbose is True:
            print('Scaling mesh to original mesh... ')
            print(icp_transform)
        mesh = scale_mesh(mesh, old_mesh=path_original_mesh, scale=scale, offset=offset, icp_transform=icp_transform, verbose=verbose)

    if path_save is not None:
        mesh.save_mesh(os.path.join(path_save, filename))
    return mesh
    
    
def get_sdfs_diffusion(
    model, 
    plane_features, 
    samples, 
    latent_vector, 
    batch_size=32**3,
    verbose=False
):
    
    n_pts_total = samples.shape[0]

    current_idx = 0
    batch = 0
    sdf_values = torch.zeros(samples.shape[0])
    
    fx = model.sdf_model.forward_with_plane_features
    
    while current_idx < n_pts_total:
        if (verbose is True) & (batch%10 == 0):
            print(batch)
        current_batch_size = min(batch_size, n_pts_total - current_idx)
        sampled_pts = samples[None, current_idx : current_idx + current_batch_size, :3]
        sdf_values[current_idx : current_idx + current_batch_size] = fx(
            plane_features,
            sampled_pts
        ).detach().cpu()

        current_idx += current_batch_size
        batch += 1
    
    return sdf_values