from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import torch
from torch.utils.data import Dataset
import pymskt as mskt

from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

import sys

import diffusion_net

import optuna
import wandb

from computer import computer
import optuna

# if computer == 'desktop':
#     path_df = './training_info_Mar.5.2022.csv'
#     OP_CACHE_DIR = '/home/gattia/data/notebooks/stanford/diffusion_net/data/op_cache'

#     sys.path.append("/home/gattia/programming/diffusion-net/src/")
# elif computer == 'server':
#     path_df = '/bmrNAS/people/aagatti/projects/Diffusion_Net/training_info_Feb.21.2022.csv'
#     OP_CACHE_DIR = '/bmrNAS/people/aagatti/projects/Diffusion_Net/notebooks/diffusion_net/data/op_cache'

#     sys.path.append('/bmrNAS/people/aagatti/projects/Diffusion_Net')
#     sys.path.append('/dataNAS/people/aagatti/programming/diffusion-net/src/')



# def train_diffusion_net(config, model, train_loader, test_loader, trial=None, use_wandb=False):    
#     model.to(config['device'])
#     if config['optimizer'] == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
#     elif config['optimizer'] == 'sgd':
#         optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

#     if use_wandb is True:
#         wandb.login(key='0090383c93d813be55bb603b3d59ace0a4b5df6c')
#         wandb.init(
#             # Set the project where this run will be logged
#             project=config["project_name"], # "diffusion-net-predict-sex",
#             entity=config["entity_name"], # "bone-modeling",
#             # Track hyperparameters and run metadata
#             config=config,
#             name=config['run_name'],
#             tags=config['tags']
#         )
    
#     if config['patience'] is not None:
#         best_test_acc = 0
#         patience_steps = 0

#     if config['scheduler'] == 'cosine_anneal':
#         config['scheduler_'] = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, 
#             T_max=config['Tmax'], 
#             eta_min=config['lr_min'], 
#             last_epoch=-1, 
#             verbose=False
#         )
#     elif config['scheduler'] == 'cosine_anneal_warm_restarts':
#         config['scheduler_'] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             optimizer, 
#             T_0=config['Tmax'], 
#             T_mult=config['Tmult'], 
#             eta_min=config['lr_min'], 
#             last_epoch=- 1, 
#             verbose=False
#         )
#     elif config['scheduler'] == 'cyclic_lr':
#         config['scheduler_'] = torch.optim.lr_scheduler.CyclicLR(
#             optimizer, 
#             base_lr=config['lr'], 
#             max_lr=config['lr_max'], 
#             step_size_up=config['cyclic_lr_step_up'], 
#             step_size_down=None, 
#             mode='triangular', 
#             gamma=1.0, 
#             scale_fn=None, 
#             scale_mode='cycle', 
#             cycle_momentum=True, 
#             base_momentum=0.8, 
#             max_momentum=0.9, 
#             last_epoch=- 1, 
#             verbose=False
#         )
#     elif config['scheduler'] == 'one_cycle_lr':
#         config['scheduler_'] = torch.optim.lr_scheduler.OneCycleLR(
#             optimizer, 
#             max_lr=config['lr_max'], 
#             total_steps=config['one_cycle_lr_total_steps'], 
#             epochs=None, 
#             steps_per_epoch=None, 
#             pct_start=0.3, 
#             anneal_strategy='cos', 
#             cycle_momentum=True, 
#             base_momentum=0.85, 
#             max_momentum=0.95, 
#             div_factor=25.0, 
#             final_div_factor=10000.0, 
#             three_phase=False, 
#             last_epoch=- 1, 
#             verbose=False
#         )



#     for epoch in range(config['n_epochs']):
#         train_acc, train_loss = train_epoch(model, train_loader, optimizer=optimizer, config=config, epoch=epoch, return_loss=True)
#         test_acc, test_loss = test(model, test_loader, config=config, return_loss=True)
#         print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))
        
#         if trial is not None:
#             # report epoch test_accuracy
#             trial.report(test_acc, epoch)
            
#             # Handle pruning based on the intermediate value.
#             if trial.should_prune():
#                 raise optuna.exceptions.TrialPruned()
        
#         # Step the scheulder
#         if config['scheduler'] in ('cosine_anneal', 'cosine_anneal_warm_restarts'):
#             config['scheduler_'].step()

#         if use_wandb is True:
#             wandb.log({
#                 "train_acc": train_acc,
#                 "train_loss": train_loss,
#                 "test_acc": test_acc,
#                 "test_loss": test_loss
#             })
        
#         if config['patience'] is not None:
#             if test_acc > best_test_acc:
#                 best_test_acc = test_acc
#                 patience_steps = 0
#             else:
#                 patience_steps += 1

#             if config['warmup_patience'] is not None:
#                 # if we want warmup, and not done, skip rest of the loop. 
#                 if epoch < config['warmup_patience']:
#                     continue
#             if patience_steps >= config['patience']:
#                 break
        
        
        
#     return test_acc
    
    