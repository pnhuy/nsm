import torch

from GenerativeAnatomy.sdf.mesh import create_mesh_diffusion_sdf

from GenerativeAnatomy.sdf.datasets import (
    read_mesh_get_sampled_pts, 
    read_meshes_get_sampled_pts
)

try:
    from pytorch3d.loss import chamfer_distance
    __chamfer__ = True
except:
    print('Error importing `chamfer_distance` from pytorch3d.loss')
    __chamfer__ = False

try:
    from GenerativeAnatomy.dependencies import sinkhorn
    __emd__ = True
except:
    print('Error importing `sinkhorn` from GenerativeAnatomy.dependencies')
    __emd__ = False


def reconstruct_mesh_diffusion_sdf(
    model,
    path,
    latent_size=256,
    calc_symmetric_chamfer=False,
    calc_emd=False,
    return_unscaled=True,
    n_pts_per_axis=256,
    return_latent=False,
    register_similarity=True,
    n_pts_per_axis_mean_mesh=128,
    scale_all_meshes=True, #whether when scaling a model down the scale should be on all points in all meshes or not. 
    mesh_to_scale=0,
    scale_method='max_rad',
    n_encode_iterations=100, #randomly sample from pc and encode - use mean encoded latent to decode 
    point_cloud_size=1024,
    scale_to_original_mesh=True,
    verbose=False
    
    # Parameters used for fitting SDF to data
    # Could use in future as a fine-tuning after inference performed using
    # the encoder? 
#     num_iterations=1000
#     lr=5e-4
#     loss_weight=1.0
#     loss_type='l1'
#     l2reg=False
#     latent_init_std=0.01
#     latent_init_mean=0.0
#     clamp_dist=None
#     latent_reg_weight=1e-4
#     n_lr_updates=2
#     lr_update_factor=10
#     convergence='num_iterations'
#     convergence_patience=50
):
    if isinstance(path, str):
        multi_object = False
    elif isinstance(path, (list, tuple)):
        multi_object = True

    # if not isinstance(decoders,(list, tuple)):
    #     decoders = [decoders,]
    if register_similarity is True:
        # create mean mesh of only mesh, or "mesh_to_scale" if more than one. 
        mean_latent = torch.zeros(1, latent_size*3)
        mean_mesh = create_mesh_diffusion_sdf(
            model, 
            latent_vector=mean_latent.cuda(),
            n_pts_per_axis=n_pts_per_axis_mean_mesh,
            verbose=verbose,
        )

        if mean_mesh is None:
            # Mean mesh is None if the zero latent vector is not well defined/learned
            # yet. In this case, the results will be very poor, might as well skip. 
            result = {
                'mesh': [None, ]
            }
            if calc_symmetric_chamfer:
                for idx in range(len(decoders)):
                    result[f'chamfer_{idx}'] = np.nan
            if calc_emd:
                for idx in range(len(decoders)):
                    result['emd_{idx}'] = np.nan
            if return_latent:
                result['latent'] = mean_latent
            return result
    if multi_object is False:
        result_ = read_mesh_get_sampled_pts(
            path, 
            center_pts=True,
            #TODO: Add axis align back in - see code commented above for example
            # axis_align=False,
            norm_pts= True,
            scale_method=scale_method,
            get_random=False,
            return_orig_mesh=True if (calc_symmetric_chamfer & return_unscaled) else False, # if want to calc, then need orig mesh
            return_new_mesh=True if (calc_symmetric_chamfer & (return_unscaled==False)) else False,
            return_orig_pts=True if (calc_symmetric_chamfer & return_unscaled) else False,
            register_to_mean_first=True if register_similarity else False,
            mean_mesh=mean_mesh if register_similarity else None,
        )
    elif multi_object is True:
        raise NotImplementedError('Multiple Objects are note yet testes for diffusion sdf!')
        result_ = read_meshes_get_sampled_pts(
            paths=path,
            mean=[0,0,0], 
            center_pts=True,
            #TODO: Add axis align back in - see code commented above for example
            # axis_align=False,
            norm_pts=True,
            scale_all_meshes=scale_all_meshes,
            mesh_to_scale=mesh_to_scale,
            scale_method=scale_method,
            get_random=False,
            return_orig_mesh=True if (calc_symmetric_chamfer & return_unscaled) else False, # if want to calc, then need orig mesh
            return_new_mesh=True if (calc_symmetric_chamfer & (return_unscaled==False)) else False,
            return_orig_pts=True if (calc_symmetric_chamfer & return_unscaled) else False,
            register_to_mean_first=True if register_similarity else False,
            mean_mesh=mean_mesh if register_similarity else None,
        )
    
    xyz = result_['pts']
    sdf_gt = result_['sdf']

    if not isinstance(xyz, torch.Tensor):
        xyz = torch.from_numpy(xyz).float()
    if multi_object is True:
        for sdf_idx, sdf_gt_ in enumerate(sdf_gt):
            if not isinstance(sdf_gt_, torch.Tensor):
                sdf_gt[sdf_idx] = torch.from_numpy(sdf_gt_).float()

            if len(sdf_gt[sdf_idx].shape) == 1:
                sdf_gt[sdf_idx] = sdf_gt[sdf_idx].unsqueeze(1)
    elif multi_object is False:
        if not isinstance(sdf_gt, torch.Tensor):
            sdf_gt = torch.from_numpy(sdf_gt).float()

        if len(sdf_gt.shape) == 1:
            sdf_gt = sdf_gt.unsqueeze(1)

    latent_mu = torch.zeros(n_encode_iterations, latent_size*3)
#     latent_std = torch.zeros(100, latent_size*3)
    for iter_ in range(n_encode_iterations):
        pt_cloud_perm = torch.randperm(xyz.size(0))
        idx_pc = pt_cloud_perm[:point_cloud_size]
        pc = xyz[None, idx_pc, :].cuda()
        plane_features = model.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        mu, log_var = model.vae_model.encode(original_features)
        latent_mu[iter_, :] = mu
#         std = torch.exp(0.5 * log_var)
#         latent_std[iter_, :] = std
    
    mean_latent_mu = latent_mu.mean(axis=0)
    
    mesh = create_mesh_diffusion_sdf(
        model, 
        latent_vector=mean_latent_mu.cuda(),
        n_pts_per_axis=n_pts_per_axis,
        path_original_mesh=None,
        offset=result_['center'],
        scale=result_['scale'],
        icp_transform=result_['icp_transform'],
        R=None,
        t=None,
        s=None,
        scale_to_original_mesh=scale_to_original_mesh,
        verbose=verbose,
    )
    
    #TODO: UPDATE IN THE FUTURE TO HAVE DIFFUSION SDF PREDICT MULTIPLE SURFACES
    
    meshes = [mesh,]

    if calc_symmetric_chamfer or calc_emd:
        pts_recon = []
        xyz_recon = []
        xyz_orig = []
        for mesh_idx, mesh in enumerate(meshes):
            if mesh is not None:
                pts_recon.append(mesh.point_coords)
                xyz_recon.append(torch.from_numpy(pts_recon[-1]).float())
            else:
                pts_recon.append(None)
                xyz_recon.append(None)
            
            if multi_object is True:
                xyz_orig_ = result_['orig_pts'][mesh_idx]
            else:
                xyz_orig_ = result_['orig_pts']
            if not isinstance(xyz_orig_, torch.Tensor):
                xyz_orig_ = torch.from_numpy(xyz_orig_).float()
            xyz_orig.append(xyz_orig_)    

    if calc_symmetric_chamfer:
        if __chamfer__ is True:
            chamfer_loss = []
            for pts_idx in range(len(meshes)):
                if pts_recon[pts_idx] is None:
                    chamfer_loss.append(np.nan)
                else:
                    chamfer_loss_, _ = chamfer_distance(xyz_orig[pts_idx].unsqueeze(0), xyz_recon[pts_idx].unsqueeze(0))
                    chamfer_loss.append(chamfer_loss_.item())
            if verbose is True:
                print('general xyz mean, min, max', torch.mean(xyz, dim=0), torch.min(xyz, dim=0), torch.max(xyz, dim=0))
                print('orig mean, min, max', torch.mean(xyz_orig, dim=0), torch.min(xyz_orig, dim=0), torch.max(xyz_orig, dim=0))
                print('new mean, min, max', torch.mean(xyz_recon, dim=0), torch.min(xyz_recon, dim=0), torch.max(xyz_recon, dim=0))
        elif __chamfer__ is False:
            raise ImportError('Cannot calculate symmetric chamfer distance without chamfer_pytorch module')
    
    if calc_emd:
        if __emd__ is True:
            emd_loss = []
            for pts_idx in range(len(meshes)):
                if pts_recon[pts_idx] is None:
                    emd_loss.append(np.nan)
                else:
                    emd_loss_, _, _ = sinkhorn(xyz_orig[pts_idx], xyz_recon[pts_idx])
                    emd_loss.append(emd_loss_.item())
        elif __emd__ is False:
            raise ImportError('Cannot calculate EMD without emd module') 
    
    if calc_emd or calc_symmetric_chamfer or return_latent:
        result = {
            'mesh': meshes,
        }
        if calc_symmetric_chamfer:
            for idx, chamfer_loss in enumerate(chamfer_loss):
                result[f'chamfer_{idx}'] = chamfer_loss
        if calc_emd:
            for idx, emd_loss in enumerate(emd_loss):
                result['emd_{idx}'] = emd_loss
        if return_latent:
            result['latent'] = mean_latent_mu
        return result
    else:
        return meshes


