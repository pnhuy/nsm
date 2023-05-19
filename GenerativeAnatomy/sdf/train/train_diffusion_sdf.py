import time
import torch
import wandb
import os

from GenerativeAnatomy.sdf.sdf_utils import (
    save_model,
    get_checkpoints,
)

from GenerativeAnatomy.sdf.train.utils import (
    add_plain_lr_to_config
)

from GenerativeAnatomy.sdf.reconstruct import get_mean_errors


def train_diffusion_sdf(
    config,
    model,
    sdf_dataset,
    use_wandb=False,
):
    model = model.to(config['device'])

    config = add_plain_lr_to_config(config)
    config['checkpoints'] = get_checkpoints(config)

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
        wandb.watch(model, log='all')
    

    data_loader = torch.utils.data.DataLoader(
        sdf_dataset,
        batch_size=config['objects_per_batch'],
        shuffle=True,
        num_workers=config['num_data_loader_threads'],
        drop_last=True if config['objects_per_batch'] < len(sdf_dataset) else False,
        # pin_memory=True,
        # persistent_workers=True,
    )

    # configure the optimizers 
    optimizer = model.configure_optimizers()

    for epoch in range(config['n_epochs']):
        # print('Epoch: ', epoch)
        start = time.time()
        model.train()
        torch.set_grad_enabled(True)
        log_dict = {}
        steps = 0
        for batch_idx, (sdf_data_batch, indices) in enumerate(data_loader):
            # print('\tBatch: ', batch_idx)
            optimizer.zero_grad()
            # the training step also does the backward pass
            log_dict_ = model.training_step(sdf_data_batch, batch_idx)
            if log_dict_ is None:
                print('\t\tSkipping batch: ', batch_idx)
                # skip this batch
                continue
            optimizer.step()
            for key, value in log_dict_.items():
                if key in log_dict:
                    log_dict[key] += value
                else:
                    log_dict[key] = value
            steps += 1
        end = time.time()
        seconds_elapsed = end - start
        
        for key, value in log_dict.items():
            log_dict[key] = value / steps
        
        if epoch in config['checkpoints']:
            save_model(
                config=config,
                epoch=epoch,
                decoder=model,
            )

            #TODO: Maybe add an option for when/how often validation is performed?
            if ('val_paths' in config) & (config['val_paths'] is not None):
                dict_loss = get_mean_errors(
                    mesh_paths=config['val_paths'],
                    num_iterations=config['num_iterations_recon'],
                    decoders=model,
                    # num_iterations=100,
                    latent_size=config['latent_size'],
                    calc_symmetric_chamfer=config['chamfer'],
                    calc_emd=config['emd'],
                    model_type='diffusion',
                    point_cloud_size=config['point_cloud_size'],
                    register_similarity=True,
                    verbose=config['verbose'],
                    get_rand_pts=config['get_rand_pts_recon'],
                    n_pts_random=config['n_pts_random_recon'],
                    lr=config['lr_recon'],
                    l2reg=config['l2reg_recon'],
                    clamp_dist=config['clamp_dist_recon'],
                    n_lr_updates=config['n_lr_updates_recon'],
                    lr_update_factor=config['lr_update_factor_recon'],
                    convergence_patience=config['convergence_patience_recon'],
                    batch_size_latent_recon=config['batch_size_latent_recon'],
                    convergence=config['convergence_type_recon'],
                    sigma_rand_pts=config['sigma_rand_pts_recon'],
                    n_samples_latent_recon=config['n_samples_latent_recon'], 
                )
                for key, value in dict_loss.items():
                    log_dict[key] = value

        
        log_dict['epoch_time_s'] = seconds_elapsed

        if use_wandb is True:
            wandb.log(log_dict)
    
    return



