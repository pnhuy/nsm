import torch
import wandb
from .utils import setup_scheduler, setup_optimizer, update_config_param
from .train_test import train_epoch, test
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np

def train_diffusion_net(config, model, train_loader, test_loader, trial=None, use_wandb=False):    
    model.to(config['device'])
    optimizer = setup_optimizer(config, model)

    if use_wandb is True:
        wandb.login(key='0090383c93d813be55bb603b3d59ace0a4b5df6c')
        wandb.init(
            # Set the project where this run will be logged
            project=config["project_name"], # "diffusion-net-predict-sex",
            entity=config["entity_name"], # "bone-modeling",
            # Track hyperparameters and run metadata
            config=config,
            name=config['run_name'],
            tags=config['tags']
        )
        wandb.watch(model, log='all')
    
    best_test_acc = 0
    patience_steps = 0

    # attache scheduler to config file. 
    scheduler_ = setup_scheduler(config, optimizer)
    config['scheduler_'] = scheduler_

    for epoch in range(config['n_epochs']):
        if config['model_type'] in ('vae', 'vanilla_vae', 'pointnet'):
            if config['verbose'] is True:
                print('batch size: ', config['batch_size'])
            train_loss, train_recon_loss, train_kld_loss, train_r2_loss = train_epoch(model, train_loader, optimizer=optimizer, config=config, epoch=epoch, return_loss=True)
            test_loss, test_recon_loss, test_kld_loss, test_r2_loss, pred_pts, true_pts, test_latents, test_diff = test(model, test_loader, config=config, return_loss=True)
            test_acc = -test_loss
            train_acc = -train_loss
            # print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))
        else:
            train_acc, train_loss = train_epoch(model, train_loader, optimizer=optimizer, config=config, epoch=epoch, return_loss=True)
            test_acc, test_loss = test(model, test_loader, config=config, return_loss=True)
            print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))

        # STUFF FOR OPTUNA
        if trial is not None:
            # report epoch test_accuracy
            trial.report(test_acc, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Step the scheduler - make sure that it handles stepping appropriately (regardless of whether or not cooldown is set.)
        if 'lr_cooldown' in config:
            cooldown_patience = config['lr_cooldown_patience']
            cooldown_start = config['n_epochs'] - cooldown_patience
            if (config['lr_cooldown'] is True) & (epoch > cooldown_start):
                if config['verbose'] is True:
                    print('Setting LR to be cooldown rate - AFTER EPOCH')
                config['lr'] = optimizer.param_groups[0]['lr']
                config['lr'] = update_config_param(config, 'lr', epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['lr']
            elif config['scheduler'] in ('cosine_anneal', 'cosine_anneal_warm_restarts'):
                config['scheduler_'].step()
        elif config['scheduler'] in ('cosine_anneal', 'cosine_anneal_warm_restarts'):
                config['scheduler_'].step()
                
        if config['verbose'] is True:
            print('LEARNING RATE INFORMATION FOR THE EPOCH') 
            print('config lr', config['lr'])
            print('optimizer lr', optimizer.param_groups[0]['lr'])
        
        # IF WE HAVE A CYCLIC KLD DEFINED THEN UPDATE KLD_WEIGHT
        if 'kld_weight_cyclic_anneal_linear' in config:
            if config['kld_weight_cyclic_anneal_linear'] is True:
                config['kld_weight'] = update_config_param(config, 'kld_weight', epoch)
            else:
                pass
        # IF WE HAVE A CYCLIC ASSD_PROPORTION THEN UPDATE IT
        if 'assd_proportion_cyclic_anneal_linear' in config:
            if config['assd_proportion_cyclic_anneal_linear'] is True:
                config['assd_proportion'] = update_config_param(config, 'assd_proportion', epoch)
            else:
                pass


        if use_wandb is True:
            if config['project_name'] in (
                "predict-sex-from-bone",
                "predict-kl-from-bone"
            ):
                log_dict = {
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "test_acc": test_acc,
                    "test_loss": test_loss
                }
                if config['log_lr_per_epoch'] is True:
                    log_dict['lr_batch'] = optimizer.param_groups[0]['lr']
                wandb.log(log_dict)
            elif config['model_type'] in ('vae', 'vanilla_vae', 'pointnet'):
                log_dict = {
                    "train_loss": train_loss, 
                    "train_recon_loss": train_recon_loss, 
                    "train_kld_loss": train_kld_loss, 
                    "train_r2_loss": train_r2_loss,

                    "test_loss": test_loss, 
                    "test_recon_loss": test_recon_loss, 
                    "test_kld_loss": test_kld_loss, 
                    "test_r2_loss": test_r2_loss,
                }

                if config['log_pt_clouds'] is True:
                    if config['colour_pt_cloud'] is True:
                        cmap = cm.viridis
                        norm = Normalize(vmin=config['pt_cloud_colour_min'], vmax=config['pt_cloud_colour_max'])
                    for pt_cloud_idx in range(config['n_pt_clouds_log']):
                        # print(type(pred_pts[pt_cloud_idx]))
                        # print(pred_pts[pt_cloud_idx].shape)
                        
                        preds_ = pred_pts[pt_cloud_idx]
                        true_ = true_pts[pt_cloud_idx]
                        if config['verbose'] is True:
                            print('pred_pts shape', preds_.shape)
                            print('true_pts shape', true_.shape)

                        if config['colour_pt_cloud'] is True:
                            diff_ = test_diff[pt_cloud_idx]
                            if config['verbose'] is True:
                                print('diff_ shape', diff_.shape)
                            if config['norm_pt_cloud_colour'] is True:
                                diff_ = norm(diff_)
                            
                            colours = cmap(diff_) * 255
                            if config['verbose'] is True:
                                print('colours shape: ', colours.shape)
                                print('preds_ shape: ', preds_.shape)
                                print('true_ shape: ', true_.shape)
                            preds_ = np.concatenate((preds_, colours[:,:3]), axis=1)
                            true_ = np.concatenate((true_, colours[:,:3]), axis=1)

                            # create colorized versions of each mesh and combine to one mesh for duel visualizations
                            true_red = np.zeros_like(true_)
                            true_red[:,:3] = true_[:,:3]
                            true_red[:,3] = 255
                            pred_blue = np.zeros_like(preds_)
                            pred_blue[:,:3] = preds_[:,:3]
                            pred_blue[:,-1] = 255
                            both_color = np.concatenate((true_red, pred_blue), axis=0)
                        log_dict[f'test_points_{pt_cloud_idx}_pred'] = wandb.Object3D(preds_)
                        log_dict[f'test_points_{pt_cloud_idx}_true'] = wandb.Object3D(true_)
                        log_dict[f'test_points_{pt_cloud_idx}_true_red_pred_blue'] = wandb.Object3D(both_color)
                        

                if config['log_latent_hist'] is True:
                    for hist_idx in range(min(config['n_latent_hist_log'], len(test_latents))):
                        values = np.nan_to_num(test_latents[hist_idx], nan=0.0, posinf=100, neginf=-100)
                        log_dict[f"test_latents_{hist_idx}"] = wandb.Histogram(values)

                if config['log_lr_per_epoch'] is True:
                    log_dict['lr_batch'] = optimizer.param_groups[0]['lr']
                
                if config['log_kld_weight'] is True:
                    log_dict['kld_weight_batch'] = config['kld_weight']

                if config['log_assd_proportion'] is True:
                    log_dict['assd_proportion_batch'] = config['assd_proportion']
                wandb.log(log_dict)
            else:
                raise NotImplementedError(f'The project: {config["project_name"]} is not setup yet!')
                # IF YOU GOT THE ABOVE ERROR - SETUP THE APPROPRIATE wandb.log({}) COMMAND FOR THE NEW
                # PROJECT. THIS PROJECT USED ACC/LOSS TO BE LOGGED. IT MIGHT BE THAT SOME OTHER METRIC(S)
                # SHOULD BE USED FOR THIS PROJECT. 
 
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_steps = 0
        else:
            patience_steps += 1

        # if we want warmup, and not done, skip rest of the loop. 
        if epoch < config['warmup_patience']:
            continue
        elif patience_steps >= config['patience']:
            # only if the warmup is done, then assess if the patience has run out
            break
        else:
            pass

        
    return test_acc
    
    