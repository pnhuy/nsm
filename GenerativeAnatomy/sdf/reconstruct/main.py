import torch
from GenerativeAnatomy.sdf.datasets import read_mesh_get_sampled_pts
from GenerativeAnatomy.sdf.mesh import create_mesh
# from GenerativeAnatomy.sdf.testing import get_mean_errors
import numpy as np
import sys
import os
import pymskt as mskt
import wandb

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

# Update LR
def adjust_learning_rate(
    initial_lr, optimizer, iteration, decreased_by, adjust_lr_every
):
    lr = initial_lr * ((1 / decreased_by) ** (iteration // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def reconstruct_latent(
    decoder, 
    num_iterations,
    latent_size,
    new_sdf, #Nx4
    loss_type='l1',
    lr=5e-4,
    loss_weight=1.0,
    l2reg=False,
    latent_init_std=0.01,
    latent_init_mean=0.0,
    clamp_dist=None,
    latent_reg_weight=1e-4,
    n_lr_updates=2,
    lr_update_factor=10,
    convergence='num_iterations',
    convergence_patience=50,
    log_wandb=False,
):

    if convergence != 'num_iterations':
        num_iterations = max(num_iterations, 10000)
    
    adjust_lr_every = num_iterations // n_lr_updates

    # Parse the inputted SDF
    if type(new_sdf) is str:
        # if string, assume path to mesh
        #TODO: Do we want to add option to sample points in/out of mesh?
        #      Will this speed up convergence?
        result_ = read_mesh_get_sampled_pts(
            new_sdf, 
            get_random=False, 
            center_pts=True, 
            norm_pts=True, 
            scale_method='max_rad',
            return_orig_mesh=False,
            return_new_mesh=False,
            return_orig_pts=False
        )
        xyz = result_['pts'] 
        sdf_gt = result_['sdf']
        xyz = torch.from_numpy(xyz).float()
        sdf_gt = torch.from_numpy(sdf_gt).float().unsqueeze(1)
    elif isinstance(new_sdf, (torch.Tensor, np.ndarray)):
        if new_sdf.shape[1] == 3:
            # if 
            xyz = new_sdf
            sdf_gt = torch.zeros(xyz.shape[0], 1)
        elif new_sdf.shape[1] == 4:
            xyz = new_sdf[:, :3]
            sdf_gt = new_sdf[:, 3].unsqueeze(1)
        else:
            raise ValueError(f'Inputted SDF must have shape Nx3 or Nx4 got: {new_s}')

    n_samples = xyz.shape[0]
    
    if clamp_dist is not None:
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
    sdf_gt = sdf_gt.cuda()
    # Figure out information about LR updates

    # Initialize latent vector
    latent = torch.ones(1, latent_size).normal_(mean=latent_init_mean, std=latent_init_std)
    latent.requires_grad = True

    # Initialize optimizer
    optimizer = torch.optim.Adam([latent], lr=lr)

    # Initialize loss
    loss = 0
    if loss_type == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif loss_type == 'l2':
        loss_fn = torch.nn.MSELoss()

    # Initialize convergence tracking
    patience = 0
    loss = 10 

    for step in range(num_iterations):
        decoder.cuda()
        decoder.eval()

        #update LR
        adjust_learning_rate(
            initial_lr=lr,
            optimizer=optimizer,
            iteration=step, 
            decreased_by=lr_update_factor, 
            adjust_lr_every=adjust_lr_every
        )

        optimizer.zero_grad()

        latent_input = latent.expand(n_samples, -1)
        inputs = torch.cat([latent_input, xyz], dim=1).cuda()

        pred_sdf = decoder(inputs)

        if clamp_dist is not None:
            pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
        
        _loss_ = loss_fn(pred_sdf, sdf_gt) * loss_weight

        if l2reg is True:
            latent_loss_ = latent_reg_weight * torch.mean(latent ** 2)
            loss_ = _loss_ + latent_loss_
        else:
            loss_ = _loss_

        loss_.backward()
        optimizer.step()

        if step % 50 == 0:
            print('Step: ', step, 'Loss: ', loss_.item())
            print('\tLatent norm: ', latent.norm)
        
        if log_wandb is True:
            wandb.log({
                'l1_loss': loss_.item(),
                'recon_loss': _loss_.item(),
                'latent_loss': latent_loss_.item(),
                'latent_norm': latent.norm().item()
            })

        if convergence == 'overall_loss':
            if loss_ < loss:
                loss = loss_
                latent_ = torch.clone(latent)
                patience = 0
            else:
                patience += 1
            
            if patience > convergence_patience:
                print('Converged!')
                break
        elif convergence == 'recon_loss':
            raise Exception('Not implemented yet')
        else:
            loss = loss_
            latent_ = torch.clone(latent)
            
    return loss, latent_

def reconstruct_mesh(
    path,
    decoder,
    latent_size,
    num_iterations=1000,
    lr=5e-4,
    loss_weight=1.0,
    loss_type='l1',
    l2reg=False,
    latent_init_std=0.01,
    latent_init_mean=0.0,
    clamp_dist=None,
    latent_reg_weight=1e-4,
    n_lr_updates=2,
    lr_update_factor=10,
    calc_symmetric_chamfer=False,
    calc_emd=False,
    return_unscaled=True,
    n_pts_per_axis=256,
    log_wandb=False,
    return_latent=False,
    convergence='num_iterations',
    convergence_patience=50,
):
    result_ = read_mesh_get_sampled_pts(
        path, 
        center_pts=True,
        #TODO: Add axis align back in - see code commented above for example
        # axis_align=False,
        norm_pts=True,
        scale_method='max_rad',
        get_random=False,
        return_orig_mesh=True if (calc_symmetric_chamfer & return_unscaled) else False, # if want to calc, then need orig mesh
        return_new_mesh=True if (calc_symmetric_chamfer & (return_unscaled==False)) else False,
        return_orig_pts=True if (calc_symmetric_chamfer & return_unscaled) else False,
        return_center=return_unscaled,
        return_scale=return_unscaled,
    )

    xyz = result_['pts']
    sdf_gt = result_['sdf']
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.from_numpy(xyz).float()
    if not isinstance(sdf_gt, torch.Tensor):
        sdf_gt = torch.from_numpy(sdf_gt).float()
    
    if len(sdf_gt.shape) == 1:
        sdf_gt = sdf_gt.unsqueeze(1)

    new_sdf = torch.cat([xyz, sdf_gt], dim=1)

    loss, latent = reconstruct_latent(
        decoder=decoder, 
        num_iterations=num_iterations,
        latent_size=latent_size,
        new_sdf=new_sdf, #Nx4
        lr=lr,
        loss_weight=loss_weight,
        loss_type=loss_type,
        l2reg=l2reg,
        latent_init_std=latent_init_std,
        latent_init_mean=latent_init_mean,
        clamp_dist=clamp_dist,
        latent_reg_weight=latent_reg_weight,
        n_lr_updates=n_lr_updates,
        lr_update_factor=lr_update_factor,
        log_wandb=log_wandb,
        convergence=convergence,
        convergence_patience=convergence_patience,
    )

    mesh = create_mesh(
        decoder=decoder.cuda(),
        latent_vector=latent.cuda(),
        n_pts_per_axis=n_pts_per_axis,
        path_original_mesh=path
    )

    if calc_symmetric_chamfer or calc_emd:
        pts_recon = mskt.mesh.meshTools.get_mesh_physical_point_coords(mesh)
        xyz_recon = torch.from_numpy(pts_recon).float()

    if calc_symmetric_chamfer:
        if __chamfer__ is True:
            xyz_orig = result_['orig_pts']
            if not isinstance(xyz_orig, torch.Tensor):
                xyz_orig = torch.from_numpy(xyz_orig).float()

            # print('general xyz mean, min, max', torch.mean(xyz, dim=0), torch.min(xyz, dim=0), torch.max(xyz, dim=0))
            # print('orig mean, min, max', torch.mean(xyz_orig, dim=0), torch.min(xyz_orig, dim=0), torch.max(xyz_orig, dim=0))
            # print('new mean, min, max', torch.mean(xyz_recon, dim=0), torch.min(xyz_recon, dim=0), torch.max(xyz_recon, dim=0))

            chamfer_loss, _ = chamfer_distance(xyz_orig.unsqueeze(0), xyz_recon.unsqueeze(0))
        
        elif __chamfer__ is False:
            raise ImportError('Cannot calculate symmetric chamfer distance without chamfer_pytorch module')
    
    if calc_emd:
        if __emd__ is True:
            xyz_orig = result_['orig_pts']
            if not isinstance(xyz_orig, torch.Tensor):
                xyz_orig = torch.from_numpy(xyz_orig).float()

            emd_loss, _, _ = sinkhorn(xyz_orig, xyz_new)
        
        elif __emd__ is False:
            raise ImportError('Cannot calculate EMD without emd module') 
    
    if calc_emd or calc_symmetric_chamfer or return_latent:
        result = {
            'mesh': mesh,
        }
        if calc_symmetric_chamfer:
            result['chamfer'] = chamfer_loss.item()
        if calc_emd:
            result['emd'] = emd_loss.item()
        if return_latent:
            result['latent'] = latent
        return result
    else:
        return mesh

def tune_reconstruction(
    model,
    config,     
    use_wandb=True
):
    if use_wandb is True:
        wandb.login(key=os.environ['WANDB_KEY'])
        wandb.init(
            # Set the project where this run will be logged
            project=config["project_name"], # "diffusion-net-predict-sex",
            entity=config["entity_name"], # "bone-modeling",
            # Track hyperparameters and run metadata
            config=config,
            name=config['run_name'],
            tags=config['tags']
        )
    
    dict_loss = get_mean_errors(
        mesh_paths=config['mesh_paths'],
        decoder=model,
        num_iterations=config['num_iterations'],
        latent_size=config['latent_size'],
        lr=config['lr'],
        loss_weight=config['loss_weight'],
        loss_type=config['loss_type'],
        l2reg=config['l2reg'],
        latent_init_std=config['latent_init_std'],
        latent_init_mean=config['latent_init_mean'],
        clamp_dist=config['clamp_dist'],
        latent_reg_weight=config['latent_reg_weight'],
        n_lr_updates=config['n_lr_updates'],
        lr_update_factor=config['lr_update_factor'],
        calc_symmetric_chamfer=config['chamfer'],
        calc_emd=config['emd'],
        convergence=config['convergence'],
        convergence_patience=config['convergence_patience'],
        log_wandb=True,
    )

    if use_wandb is True:                    
            wandb.log(
                dict_loss
            )

def get_mean_errors(
    mesh_paths,
    decoder,
    latent_size,
    calc_symmetric_chamfer=False,
    calc_emd=False,
    log_wandb=False,
    num_iterations=1000,
    lr=5e-4,
    loss_weight=1.0,
    loss_type='l1',
    l2reg=False,
    latent_init_std=0.01,
    latent_init_mean=0.0,
    clamp_dist=None,
    latent_reg_weight=1e-4,
    n_lr_updates=2,
    lr_update_factor=10,
    convergence='num_iterations',
    convergence_patience=50,
    config=None,
):
    loss = {}
    if calc_symmetric_chamfer:
        chamfer = []
    if calc_emd:
        emd = []

    for idx, mesh_path in enumerate(mesh_paths):
        result_ = reconstruct_mesh(
            path=mesh_path,
            decoder=decoder,
            latent_size=latent_size,
            calc_symmetric_chamfer=calc_symmetric_chamfer,
            calc_emd=calc_emd,
            log_wandb=log_wandb,
            num_iterations=num_iterations,
            lr=lr,
            loss_weight=loss_weight,
            loss_type=loss_type,
            l2reg=l2reg,
            latent_init_std=latent_init_std,
            latent_init_mean=latent_init_mean,
            clamp_dist=clamp_dist,
            latent_reg_weight=latent_reg_weight,
            n_lr_updates=n_lr_updates,
            lr_update_factor=lr_update_factor,
            convergence=convergence,
            convergence_patience=convergence_patience,
        )
        print('result_', result_)
        if calc_symmetric_chamfer:
            if idx == 0:
                loss['chamfer'] = []
            loss['chamfer'].append(result_['chamfer'])
        if calc_emd:
            if idx == 0:
                loss['emd'] = []
            loss['emd'].append(result_['emd'])
    
    for key, item in loss.items():
        loss[key] = np.mean(item)

    return loss

