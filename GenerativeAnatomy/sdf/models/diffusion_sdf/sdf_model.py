# https://github.com/princeton-computational-imaging/Diffusion-SDF/blob/64dc177a43517d17077e6ff7d16160cf64e78255/train_sdf/models/sdf_model.py
#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F

from einops import reduce
from .sdf_decoder import ModulatedMLP
from .conv_pointnet import ConvPointnet
# from models.archs.sdf_decoder import * 
# from models.archs.encoders.conv_pointnet import ConvPointnet


class SDFModel(nn.Module):

    def __init__(self,
        num_layers=9,
        hidden_dim=512,
        latent_dim=256,
        dropout=0.0,
        latent_input=True,
        pos_enc=False,
        skip_connection=[4],
        tanh_act=False,
        pn_hidden_dim=128,
        outer_lr=1e-4,
        plane_resolution=64,

    ):
        super(SDFModel, self).__init__()

        # self.specs = specs
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.latent_in = latent_input
        self.pos_enc = pos_enc
        self.skip_connection = skip_connection
        self.tanh_act = tanh_act
        self.pn_hidden = pn_hidden_dim
        self.outer_lr = outer_lr
        self.plane_resolution = plane_resolution

        self.pointnet = ConvPointnet(c_dim=self.latent_dim, hidden_dim=self.pn_hidden, plane_resolution=self.plane_resolution)
        
        self.model = ModulatedMLP(
            latent_size=self.latent_dim, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers, 
            dropout_prob=self.dropout, 
            latent_in=self.latent_in, 
            pos_enc=self.pos_enc,
            skip_connection=self.skip_connection, 
            tanh_act=self.tanh_act
        )
        
        self.model.train()
        #print(self.model)

        #print("encoder params: ", sum(p.numel() for p in self.pointnet.parameters() if p.requires_grad))
        #print("mlp params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(), self.outer_lr)

        return optimizer 

    def training_step(self, x, idx):

        xyz = x['xyz'].cuda() # (B, 16000, 3)
        gt = x['gt_sdf'].cuda() # (B, 16000)
        pc = x['point_cloud'].cuda() # (B, 1024, 3)

        modulations = self.pointnet(pc, xyz) 

        pred_sdf, new_mod = self.model(xyz, modulations)

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction = 'none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        return sdf_loss     

    def forward(self, modulations, xyz):
        modulations = self.pointnet(modulations, xyz)
        return self.model(xyz, modulations)[0].squeeze()

    def forward_with_plane_features(self, plane_features, xyz):
        # query/interpolate the latent features at the xyz points
        point_features = self.pointnet.forward_with_plane_features(plane_features, xyz)
        # pass the latent features + xyz coords to the sdf MLP model
        pred_sdf = self.model(xyz, point_features)[0].squeeze()
        return pred_sdf
