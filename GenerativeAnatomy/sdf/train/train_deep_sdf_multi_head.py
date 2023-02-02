from GenerativeAnatomy.sdf.sdf_utils import (
    get_learning_rate_schedules, 
    adjust_learning_rate, 
    save_latent_vectors,
    save_model,
    get_optimizer,
    get_latent_vecs,
    get_checkpoints,
)
from GenerativeAnatomy.sdf.reconstruct import get_mean_errors

import wandb
import os
import torch 
import time
import numpy as np

loss_l1 = torch.nn.L1Loss(reduction='none')

def train_deep_sdf(
    config, 
    models: tuple,
    sdf_dataset, 
    use_wandb=False):    

    config['checkpoints'] = get_checkpoints(config)
    config['lr_schedules'] = get_learning_rate_schedules(config)

    for model in models:
        model = model.to(config['device'])

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
        wandb.watch(models, log='all')


    data_loader = torch.utils.data.DataLoader(
        sdf_dataset,
        batch_size=config['objects_per_batch'],
        shuffle=True,
        num_workers=config['num_data_loader_threads'],
        drop_last=False,
    )

    latent_vecs = get_latent_vecs(len(data_loader.dataset), config)

    optimizer = get_optimizer(models, latent_vecs, config['lr_schedules'], config["optimizer"])

    for epoch in range(1, config['n_epochs'] + 1):
        # not passing latent_vecs because presumably they are being tracked by the
        # and updated in memory? 
        log_dict = train_epoch(model, data_loader, latent_vecs, optimizer=optimizer, config=config, epoch=epoch, return_loss=True)
        
        if epoch in config['checkpoints']:
            save_latent_vectors(
                config=config,
                epoch=epoch,
                latent_vec=latent_vecs,
            )
            save_model(
                config=config,
                epoch=epoch,
                decoder=models,
            )
            if ('val_paths' in config) & (config['val_paths'] is not None):
                list_dict_loss = []
                for model_idx, model in enumerate(models):
                    dict_loss = get_mean_errors(
                        mesh_paths=config['val_paths'],
                        decoder=model,
                        latent_size=config['latent_size'],
                        calc_symmetric_chamfer=config['chamfer'],
                        calc_emd=config['emd'],
                    )

                log_dict.update(dict_loss)

        if use_wandb is True:                    
            wandb.log(log_dict)
        
    return loss

def calc_weight(epoch, n_epochs, schedule, cooldown=None):
    
    if cooldown is not None:
        if epoch > (n_epochs - cooldown):
            return 1.0
        else:
            n_epochs = n_epochs - cooldown
    if schedule == 'linear':
        return epoch / n_epochs
    elif schedule == 'exponential':
        return epoch**2 / n_epochs**2
    elif schedule == 'exponential_plateau':
        return 1 - (epoch-n_epochs)**2/n_epochs**2
    elif schedule == 'constant':
        return 1.0
    else:
        raise ValueError('Unknown schedule: {}'.format(schedule))


def train_epoch(
    models, 
    data_loader,
    latent_vecs,
    optimizer,
    config, 
    epoch, 
    return_loss=True,
    verbose=False,
):
    n_surfaces = len(models)
    start = time.time()
    for model in models:
        model.train()

    adjust_learning_rate(config['lr_schedules'], optimizer, epoch)
    
    step_losses = 0
    step_l1_loss = 0
    step_code_reg_loss = 0

        # if config['code_regularization_type_prior'] == 'kld_diagonal':
        #     kld_loss = get_kld(latent_vecs)

    for sdf_data, indices in data_loader:
        if verbose is True:
            print('sdf index size:', indices.size())
            print('sdf data size:', sdf_data.size())
        
        sdf_data = sdf_data.reshape(-1, 3 + n_surfaces)

        num_sdf_samples = sdf_data.shape[0]
        sdf_data.requires_grad = False

        xyz = sdf_data[:, :3]
        assert sdf_data.shape[-1] == (n_surfaces + 3) # make sure we have one SDF value per object
        
        sdf_gt = []
        for surf_idx in range(n_surfaces):
            sdf_gt_ = sdf_data[:, 3 + surf_idx].unsqueeze(1)
            if config['enforce_minmax'] is True:
                sdf_gt_ = torch.clamp(sdf_gt_, -config['clamp_dist'], config['clamp_dist'])
            sdf_gt.append(sdf_gt_)

        xyz = torch.chunk(xyz, config['batch_split'])
        indices = torch.chunk(
            indices.unsqueeze(-1).repeat(1, config['samples_per_object_per_batch']).view(-1), #repeat the index for every sample
            config['batch_split'], # split the data into the appropriate number of batches - so can fit in ram. 
        )

        for surf_idx in range(n_surfaces):
            sdf_gt[surf_idx] = torch.chunk(sdf_gt[surf_idx], config['batch_split'])

        batch_loss = 0.0
        batch_l1_loss = 0.0
        batch_code_reg_loss = 0.0

        optimizer.zero_grad()

        # I'VE DONE UP TO HERE
        # THE sdf_gt should have SDF values for each object being jointly optimized. 
        # Now need to make sure appropriate model is applied and getting the appropriate SDFs
        # In the following. 

        for split_idx in range(config['batch_split']):
            if verbose is True:
                print('Split idx: ', split_idx)

            batch_vecs = latent_vecs(indices[split_idx])
            inputs = torch.cat([batch_vecs, xyz[split_idx]], dim=1)
            inputs = inputs.to(config['device'])

            pred_sdfs = []
            for model in models:
                pred_sdf = model(inputs)
                if config['enforce_minmax'] is True:
                    pred_sdf = torch.clamp(pred_sdf, -config['clamp_dist'], config['clamp_dist'])
                pred_sdfs.append(pred_sdf)                       

            l1_losses = []
            for surf_idx, pred_sdf in enumerate(pred_sdfs):
                l1_losses.append(loss_l1(pred_sdf, sdf_gt[surf_idx][split_idx].cuda()))
            # l1_loss = loss_l1(pred_sdf, sdf_gt[split_idx].cuda())

            if 'multi_object_overlap' in config and config['multi_object_overlap'] is True:
                raise Exception('Not implemented yet')
                # Should add some weighted penalty to the l1 loss
                # this is similar to surface_accuracy_e below
                # The idea being - the signs of the objects should never have 2 objects as (-) becuase it 
                # means one object is inside the other.
                # However, we dont want there to be any gaps between them - so we need to be intelligent about 
                # how to penalize this so that it doesnt end up with weird artefacts.
            
            # curriculum SDF equation 5
            # progressively fine-tune the regions of surface cared about by the network. 
            if config['surface_accuracy_e'] is not None:
                weight_schedule = 1 - calc_weight(epoch, config['n_epochs'], config['surface_accuracy_schedule'], config['surface_accuracy_cooldown'])
                for l1_idx, l1_loss in enumerate(l1_losses):
                    l1_losses[l1_idx] = torch.maximum(
                        l1_loss - (weight_schedule * config['surface_accuracy_e']),
                        torch.zeros_like(l1_loss)
                    )
            
            # curriculum SDF equation 6
            # progressively fine-tune the regions of surface cared about by the network.
            # weighting gives higher preference to regions closer to surface / with opposite sign. 
            if config['sample_difficulty_weight'] is not None:
                weight_schedule = calc_weight(epoch, config['n_epochs'], config['sample_difficulty_weight_schedule'], config['sample_difficulty_cooldown'])
                difficulty_weight = weight_schedule * config['sample_difficulty_weight']
                for surf_idx, surf_gt_ in enumerate(sdf_gt):
                    # Weights points independently
                    # so, if hard for one surface - then we weight it heavily, but if
                    # easy for another surface - then we weight it less.
                    error_sign = torch.sign(surf_gt_[split_idx].cuda() - pred_sdfs[surf_idx])
                    sdf_gt_sign = torch.sign(surf_gt_[split_idx].cuda())
                    sample_weights = 1 + difficulty_weight * sdf_gt_sign * error_sign
                    l1_losses[surf_idx] = l1_losses[surf_idx] * sample_weights
            elif config['sample_difficulty_lx'] is not None:
                weight_schedule = calc_weight(epoch, config['n_epochs'], config['sample_difficulty_lx_schedule'], config['sample_difficulty_lx_cooldown'])
                for surf_idx, surf_gt_ in enumerate(sdf_gt):
                    difficulty_weight = 1 / (l1_losses[surf_idx] ** config['sample_difficulty_lx']  + config['sample_difficulty_lx_epsilon'])
                    difficulty_weight = difficulty_weight * weight_schedule
                    l1_losses[surf_idx] = l1_losses[surf_idx] * difficulty_weight
            
            # Weight each surface loss by the number of samples it has
            # so that the sum of them all is the same as the mean loss. 
            for idx, l1_loss_ in enumerate(l1_losses):
                l1_losses[idx] = l1_loss_ / num_sdf_samples
            # l1_loss = l1_loss / num_sdf_samples

            # Comput total loss for each mesh (which is the same as the 
            # mean).
            l1_loss = 0
            for l1_loss_ in l1_losses:
                l1_loss += l1_loss_.sum()
            
            # Normalize by number of surfaces
            l1_loss = l1_loss / len(l1_losses)
            
            batch_l1_loss += l1_loss.item()
            chunk_loss = l1_loss

            if config['code_regularization'] is True:
                if config['code_regularization_type_prior'] == 'spherical':
                    # spherical prior
                    # all latent vectors should have the same unit length
                    # therefore, the latent dimensions will be correlated
                    # with one another - this is as opposed to PCA (and below). 
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                elif config['code_regularization_type_prior'] == 'identity':
                    # independently penalize each dimension/value of latent code
                    # therefore latent code ends up having identity covariance matrix
                    l2_size_loss = torch.sum(torch.square(batch_vecs))
                elif config['code_regularization_type_prior'] == 'kld_diagonal':
                    l2_size_loss = get_kld(batch_vecs)
                else:
                    raise ValueError('Unknown code regularization type prior: {}'.format(config['code_regularization_type_prior']))
                reg_loss = (
                    config['code_regularization_weight'] * min(1, epoch/100) * l2_size_loss
                ) / num_sdf_samples

                if config['code_cyclic_anneal'] is True:
                    anneal_weight = cyclic_anneal_linear(epoch=epoch, n_epochs=config['n_epochs'])
                    reg_loss = reg_loss * anneal_weight

                chunk_loss = chunk_loss + reg_loss.cuda()
                batch_code_reg_loss += reg_loss.item()
            
            chunk_loss.backward()

            batch_loss += chunk_loss.item()

        step_losses += batch_loss
        step_l1_loss += batch_l1_loss
        step_code_reg_loss += batch_code_reg_loss
        
        if config['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['grad_clip']
            )
        
        optimizer.step()
    end = time.time()

    seconds_elapsed = end - start

    save_loss = step_losses / len(data_loader)
    save_l1_loss = step_l1_loss / len(data_loader)
    save_code_reg_loss = step_code_reg_loss / len(data_loader)    
    print('save loss: ', save_loss)
    print('\t save l1 loss: ', save_l1_loss)
    print('\t save code loss: ', save_code_reg_loss)

    log_dict = {
        'loss': save_loss,
        'epoch_time_s': seconds_elapsed,
        'l1_loss': save_l1_loss,
        'latent_code_regularization_loss': save_code_reg_loss,
    }

    return log_dict

def cyclic_anneal_linear(
    epoch,
    n_epochs,
    min_=0,
    max_=1,
    ratio=0.5,
    n_cycles=5
):
    """
    https://github.com/haofuml/cyclical_annealing
    """
    cycle_length = np.floor(n_epochs / n_cycles).astype(int)
    cycle = epoch // cycle_length
    cycle_progress = epoch % cycle_length

    weight = (cycle_progress / cycle_length) * (1/ratio)
    weight = np.min([weight, 1])

    return min_ + (max_ - min_) * weight

def get_kld(array, samples_dim=0):
    """
    kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    Above is the KLD between a diagonal multivariate normal distribution and a standard normal distribution.
    """
    mean = torch.mean(array, dim=samples_dim)
    var = torch.var(array, dim=samples_dim)
    kld = -0.5 * torch.sum(1 + torch.log(var) - mean ** 2 - var)

    return kld
