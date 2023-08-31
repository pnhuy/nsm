# https://github.com/princeton-computational-imaging/Diffusion-SDF/blob/64dc177a43517d17077e6ff7d16160cf64e78255/train_sdf/models/combined_model.py

import torch
import torch.utils.data 
from torch.nn import functional as F
# import pytorch_lightning as pl
import torch.nn as nn

from einops import reduce

from .sdf_model import SDFModel
from .vae import BetaVAE

# add paths in model/__init__.py for new models
# from models import * 

class CombinedModel(nn.Module):
    def __init__(
        self,
        device="cuda:0",
        training_task='modulation', # 'combined' or 'modulation' or 'diffusion'
        
        latent_std=0.25, # std of target gaussian distribution of latent space
        sdf_lr=1e-4,
        vae_lr=1e-4,
        kld_weight=0.01,

        sdf_hidden_dim=512,
        sdf_latent_dim=256, # latent dim of pointnet
        sdf_num_layers=9,
        sdf_skip_connection=[4],
        pn_hidden_dim=128,

        plane_resolution=64,
       
    ):
        super(CombinedModel, self).__init__()
        # self.specs = specs

        self.device = device
        self.task = training_task
        self.latent_std = latent_std
        self.sdf_lr = sdf_lr
        self.vae_lr = vae_lr
        self.kld_weight = kld_weight

        self.sdf_latent_dim = sdf_latent_dim
        self.sdf_hidden_dim = sdf_hidden_dim
        self.sdf_num_layers = sdf_num_layers
        self.sdf_skip_connection = sdf_skip_connection
        self.pn_hidden_dim = pn_hidden_dim

        self.plane_resolution = plane_resolution


        if self.task in ('combined', 'diffusion'):
            raise NotImplementedError('Diffusion Model Not Implemented!')

        if self.task in ('combined', 'modulation'):
            self.sdf_model = SDFModel(
                num_layers=self.sdf_num_layers,
                hidden_dim=self.sdf_hidden_dim,
                latent_dim=self.sdf_latent_dim,
                pn_hidden_dim=self.pn_hidden_dim,
                plane_resolution=self.plane_resolution,
                skip_connection=self.sdf_skip_connection,
            )

            modulation_dim = self.sdf_latent_dim*3 # latent dim of modulation
            latent_std = self.latent_std # std of target gaussian distribution of latent space
            hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
            
            self.vae_model = BetaVAE(
                in_channels=self.sdf_latent_dim*3, 
                latent_dim=modulation_dim, 
                hidden_dims=hidden_dims, 
                kl_std=latent_std,
                plane_resolution=self.plane_resolution,
            )

        if self.task in ('combined', 'diffusion'):
            raise Exception('NEED diffusion model!!')
            self.diffusion_model = DiffusionModel(model=DiffusionNet(**specs["diffusion_model_specs"]), **specs["diffusion_specs"]) 
 

    def training_step(self, x, idx):

        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == 'modulation':
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        

    def configure_optimizers(self):

        if self.task == 'combined':
            raise NotImplementedError('Diffusion Model Not Implemented!')
            params_list = [
                    { 'params': list(self.sdf_model.parameters()) + list(self.vae_model.parameters()), 'lr':self.specs['sdf_lr'] },
                    { 'params': self.diffusion_model.parameters(), 'lr':self.specs['diff_lr'] }
                ]
        elif self.task == 'modulation':
            params_list = [
                    { 
                        'params': self.sdf_model.parameters(), 
                        'lr':self.sdf_lr 
                    },
                    { 
                        'params': self.vae_model.parameters(), 
                        'lr':self.vae_lr
                    }
                ]
        elif self.task == 'diffusion':
            raise NotImplementedError('Diffusion Model Not Implemented!')
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['diff_lr'] }
                ]

        optimizer = torch.optim.Adam(params_list)
        return optimizer
        # return {
        #         "optimizer": optimizer,
        #         # "lr_scheduler": {
        #         # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50000, threshold=0.0002, min_lr=1e-6, verbose=False),
        #         # "monitor": "total"
        #         # }
        # }


    #-----------different training steps for sdf modulation, diffusion, combined----------

    def train_modulation(self, x):

        xyz = x['xyz'].to(self.device) # (B, N, 3)
        gt = x['gt_sdf'].to(self.device) # (B, N)
        pc = x['point_cloud'].to(self.device) # (B, 1024, 3)

        # STEP 1: obtain reconstructed plane feature and latent code 
        # these are 3 planes where each pixel is a latent code
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        # use a VAE to compress the plane features into a latent code
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]
        # we do not use the reconstruction loss for the VAE because we dont care if
        # if re-creates the features, just if the SDF predictions are accurate/correct

        # STEP 2: pass recon back to GenSDF pipeline 
        # pass plane features to sdf model
        # query/interpolate the latent features at the xyz points
        # then pass the latent features + xyz coords to the sdf model
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.kld_weight)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        loss.backward()

        loss_dict =  {"sdf_loss": sdf_loss.item(), "vae_loss": vae_loss.item()}
        # self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss_dict


    # # currently running under this framework leads to bugs...training converges but output is garbage 
    # # will try to debug but we also welcome pull requests 
    # def train_diffusion(self, x):

    #     self.train()

    #     pc = x['point_cloud'] # (B, 1024, 3) or False if unconditional 
    #     latent = x['latent'] # (B, D)

    #     # unconditional training if cond is None 
    #     cond = pc if self.specs['diffusion_model_specs']['cond'] else None 

    #     # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
    #     # typically diff_100 approaches 0 while diff_1000 can still be relatively high
    #     # visualizing loss curves can help with debugging if training is unstable
    #     diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)

    #     loss_dict =  {
    #                     "total": diff_loss,
    #                     "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
    #                     "diff1000": diff_1000_loss
    #                 }
    #     self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

    #     return diff_loss

    # # the first half is the same as "train_sdf_modulation"
    # # the reconstructed latent is used as input to the diffusion model, rather than loading latents from the dataloader as in "train_diffusion"
    # def train_combined(self, x):
    #     xyz = x['xyz'] # (B, N, 3)
    #     gt = x['gt_sdf'] # (B, N)
    #     pc = x['point_cloud'] # (B, 1024, 3)

    #     # STEP 1: obtain reconstructed plane feature for SDF and latent code for diffusion
    #     plane_features = self.sdf_model.pointnet.get_plane_features(pc)
    #     original_features = torch.cat(plane_features, dim=1)
    #     #print("plane feat shape: ", feat.shape)
    #     out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
    #     reconstructed_plane_feature, latent = out[0], out[-1] # [B, D*3, resolution, resolution], [B, D*3]

    #     # STEP 2: pass recon back to GenSDF pipeline 
    #     pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
    #     # STEP 3: losses for VAE and SDF 
    #     try:
    #         vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
    #     except:
    #         print("vae loss is nan at epoch {}...".format(self.current_epoch))
    #         return None # skips this batch
    #     sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
    #     sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

    #     # STEP 4: use latent as input to diffusion model
    #     cond = pc if self.specs['diffusion_model_specs']['cond'] else None
    #     diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)
        
    #     # STEP 5: use predicted / reconstructed latent to run SDF loss 
    #     generated_plane_feature = self.vae_model.decode(pred_latent)
    #     generated_sdf_pred = self.sdf_model.forward_with_plane_features(generated_plane_feature, xyz)
    #     generated_sdf_loss = F.l1_loss(generated_sdf_pred.squeeze(), gt.squeeze())

    #     # surface weight could prioritize points closer to surface but we did not notice better results when using it 
    #     #surface_weight = torch.exp(-50 * torch.abs(gt))
    #     #generated_sdf_loss = torch.mean( F.l1_loss(generated_sdf_pred, gt, reduction='none') * surface_weight )

    #     # we did not experiment with using constants/weights for each loss (VAE loss is weighted using value in specs file)
    #     # results could potentially improve with a grid search 
    #     loss = sdf_loss + vae_loss + diff_loss + generated_sdf_loss

    #     loss_dict =  {
    #                     "total": loss,
    #                     "sdf": sdf_loss,
    #                     "vae": vae_loss,
    #                     "diff": diff_loss,
    #                     # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
    #                     # typically diff_100 approaches 0 while diff_1000 can still be relatively high
    #                     # visualizing loss curves can help with debugging if training is unstable
    #                     #"diff100": diff_100_loss, # note that this can sometimes appear as nan when the training batch does not have sampled timesteps < 100
    #                     #"diff1000": diff_1000_loss,
    #                     "gensdf": generated_sdf_loss,
    #                 }
    #     self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

    #     return loss