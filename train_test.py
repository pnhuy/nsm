from tqdm import tqdm
import diffusion_net
import torch
from .utils import random_scale, update_config_param, random_rotate_points
from torch import nn


def train_epoch(model, train_loader, optimizer, config, epoch, return_loss=False):
    # Implement lr decay
    if (epoch > 0) & (config['decay_rate'] != 1) & (config['scheduler'] is None) & (config['update_lr_per_batch'] is False):
        lr = config['lr'] * config['decay_rate'] ** epoch

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    steps = 0
    total_loss = 0

    if config['model_type'] in ('vae', 'vanilla_vae', 'pointnet'):
        recon_loss = 0
        kl_loss = 0
        r2_loss = 0
    else:
        correct = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):

        if (config['decay_rate'] != 1) & (config['scheduler'] is None) & (config['update_lr_per_batch'] is True):
            # calculate how many total batches have been run (epochs * batches-per-epoch + current-number-of-steps)
            n_batches = len(data)
            lr = config['lr'] * config['decay_rate'] ** (epoch * n_batches + steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr 

        # Get data
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, outcomes = data

        # Move to device
        verts = verts.to(config['device'])
        faces = faces.to(config['device'])
        frames = frames.to(config['device'])
        mass = mass.to(config['device'])
        L = L.to(config['device'])
        evals = evals.to(config['device'])
        evecs = evecs.to(config['device'])
        gradX = gradX.to(config['device'])
        gradY = gradY.to(config['device'])
        outcomes = outcomes.to(config['device'])
        
        # Randomly rotate positions
        if 'augment_random_rotate' in config:
            if config['augment_random_rotate'] is True:
                verts = random_rotate_points(verts, rotation_range_percent=config['random_rotate_range'])
                
        # Randomly scale size/positions of points (should probably only be used if the dataset is already centered).
        if 'augment_random_scale' in config:
            if config['augment_random_scale'] is True:
                verts = random_scale(
                    verts, 
                    min_=config['random_scale_min'], 
                    max_=config['random_scale_max'], 
                    center_first=config['random_scale_center_first'], 
                    log=config['random_scale_sample_in_log']
                )

        # Construct features
        if config['model_type'] not in ('vanilla_vae', 'pointnet'):
            if config['input_features'] == 'xyz':
                features = verts
            elif config['input_features'] == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        if config['model_type'] in ('vanilla_vae', 'pointnet'):
            preds = model(verts)
        else:
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        if config['model_type'] in ('vae', 'vanilla_vae', 'pointnet'):
            # Evaluate loss
            loss_dict = model.loss_function(preds, **config)
            loss_dict['loss'].backward()

            # track accuracy
            total_loss += loss_dict['loss']
            recon_loss += loss_dict['Reconstruction_Loss']
            kl_loss += loss_dict['KLD']
            r2_loss  += loss_dict['R2']
        
        # Evaluate loss
        else:
            loss = diffusion_net.utils.label_smoothing_log_loss(preds, outcomes, config['label_smoothing_frac'])
            total_loss += loss
            loss.backward()

            # track accuracy
            pred_labels = torch.max(preds, dim=-1).indices
            this_correct = pred_labels.eq(outcomes).sum().item()
            if config['verbose'] is True:
                print(f'preds: {preds}; pred_labels: {pred_labels}; outcomes: {outcomes}; this_correct: {this_correct}')
            correct += this_correct
        
        steps += 1
        
        # clip the gradients before stepping the optimizer
        if config['grad_clip_norm'] != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_norm'], norm_type=2)

        # Determine if should step optimizer
        # if config['steps_accumulate_grads'] == 1, then will update on every batch
        # default config['steps_accumulate_grads'] is 1 for this reason. 
        # otherwise, can specify config['steps_accumulate_grads'] > 1 to 
        if ((batch_idx + 1) % config['steps_accumulate_grads'] == 0) or (batch_idx + 1 == len(train_loader)):
            if config['verbose'] is True:
                print('stepping optimizer on batch iteration: ', batch_idx)
            # Step the optimizer
            optimizer.step()
            optimizer.zero_grad()
        else:
            pass

        
        # handle stepping the scheduler regardless of whether or not cooldown is set. 
        if 'lr_cooldown' in config:
            cooldown_patience = config['lr_cooldown_patience']
            cooldown_start = config['n_epochs'] - cooldown_patience
            if (config['lr_cooldown'] is True) & (epoch > cooldown_start):
                if config['verbose'] is True:
                    print('Setting LR to be cooldown rate - AFTER STEP')
                config['lr'] = optimizer.param_groups[0]['lr']
                config['lr'] = update_config_param(config, 'lr', epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['lr']
            elif config['scheduler'] in ('cyclic_lr', 'one_cycle_lr'):
                config['scheduler_'].step()
        elif config['scheduler'] in ('cyclic_lr', 'one_cycle_lr'):
                config['scheduler_'].step()
                
    if config['model_type'] in ('vae', 'vanilla_vae', 'pointnet'):
        total_loss /= steps
        recon_loss /= steps
        kl_loss /= steps
        r2_loss /= steps
        return total_loss, recon_loss, kl_loss, r2_loss

    else:
        train_acc = correct / steps
        train_loss = total_loss / steps
        
        if return_loss is False:
            return train_acc
        elif return_loss is True:
            return train_acc, train_loss

# Do an evaluation pass on the test dataset 
def test(model, test_loader, config, return_loss=False):
    
    model.eval()
    if config['model_type'] in ('vae', 'vanilla_vae', 'pointnet'):
        recon_loss = 0
        kl_loss = 0
        r2_loss = 0
        pred_pts = []
        true_pts = []
        latents = []
        diff_pts = []

    else: 
        correct = 0

    steps = 0
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, outcomes = data

            # Move to device
            verts = verts.to(config['device'])
            faces = faces.to(config['device'])
            frames = frames.to(config['device'])
            mass = mass.to(config['device'])
            L = L.to(config['device'])
            evals = evals.to(config['device'])
            evecs = evecs.to(config['device'])
            gradX = gradX.to(config['device'])
            gradY = gradY.to(config['device'])
            outcomes = outcomes.to(config['device'])
            
            # Construct features
            if (config['input_features'] == 'xyz') or (config['model_type'] in ('vanilla_vae', 'pointnet')):
                features = verts
            elif config['input_features'] == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            if config['model_type'] in ('vanilla_vae', 'pointnet'):
                preds = model(features)
            else:
                preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            # calculate loss
            
            if config['model_type'] in ('vae', 'vanilla_vae', 'pointnet'):
                # Evaluate loss
                loss_dict = model.loss_function(preds, **config)
                # track accuracy
                total_loss += loss_dict['loss']
                recon_loss += loss_dict['Reconstruction_Loss']
                kl_loss += loss_dict['KLD']
                r2_loss  += loss_dict['R2']

                if config['decoder_variance'] is True:
                    print('preds length', len(preds))
                    print('popping the variance dimension')
                    _ = preds.pop(1)
                    print('preds length', len(preds))

                # THE BELOW SHOULD BE REFORMATTED - SHOULD JUST UPDATE INDEXING IF len(preds[0].shape) ==3
                if len(preds[0].shape) == 3:
                    # just grab the first example from the batch.
                    for idx in range(preds[0].shape[0]):
                        diff_ = torch.sqrt(torch.sum(torch.square(preds[0][idx] - preds[1][idx]), dim=1))
                        diff_pts.append(diffusion_net.utils.toNP(diff_))
                        pred_pts.append(diffusion_net.utils.toNP(preds[0][idx]))
                        true_pts.append(diffusion_net.utils.toNP(preds[1][idx]))
                        latent = torch.exp(0.5 * preds[2][idx]) # convert log_var to std
                        latents.append(diffusion_net.utils.toNP(latent)) 
                else:
                    diff_ = torch.sqrt(torch.sum(torch.square(preds[0] - preds[1]), dim=1))
                    diff_pts.append(diffusion_net.utils.toNP(diff_))
                    pred_pts.append(diffusion_net.utils.toNP(preds[0]))
                    true_pts.append(diffusion_net.utils.toNP(preds[1]))
                    latent = torch.exp(0.5 * preds[2]) # convert log_var to std
                    latents.append(diffusion_net.utils.toNP(latent))
            else:
                loss = diffusion_net.utils.label_smoothing_log_loss(preds, outcomes, config['label_smoothing_frac'])
                total_loss += loss

                # track accuracy
                pred_labels = torch.max(preds, dim=-1).indices
                this_correct = pred_labels.eq(outcomes).sum().item()
                if config['verbose'] is True:
                    print(f'preds: {preds}; pred_labels: {pred_labels}; outcomes: {outcomes}; this_correct: {this_correct}')
                correct += this_correct
            steps += 1

    if config['model_type'] in ('vae', 'vanilla_vae', 'pointnet'):
        total_loss /= steps
        recon_loss /= steps
        kl_loss /= steps
        r2_loss /= steps
        return total_loss, recon_loss, kl_loss, r2_loss, pred_pts, true_pts, latents, diff_pts

    else:
        test_acc = correct / steps
        test_loss = total_loss / steps

        if return_loss is False:
            return test_acc 
        elif return_loss is True:
            return test_acc, test_loss