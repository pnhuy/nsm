import torch
from torch import nn
from .deep_sdf import Decoder
from torch.nn.functional import grid_sample




"""
We will create a triplanar neural implicit representation model. 
First, we will create a VAE that takes a latent vector, reshapes it into 
a CX2x2 tensor, and then uses a 2D CNN to output a C2xHxH tensor that is a
set of 2D planar feature maps. We will use the first 1/3 of the channels
as features for the xy plane, the second 1/3 for the xz plane, and the last
1/3 for the yz plane. 

Then, we will train an MLP as a SDF decoder. Instead of only taking the xyz 
position of each point and a fixed latent code, we will sample the latent code
from the planar feature mapes outputted from the VAE. This will be done using 
summation of the latent codes from each plane using bilinear interpolation. This 
way, we get a specific latent code for each point in space.
"""

class VAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        out_features=128*3,
        hidden_dims=[512, 512, 512, 512, 512],
        deep_image_size=2,
        norm=True,
        norm_type='batch',
        activation='leakyrelu',
        start_with_mlp=True,
    ):
        super(VAEDecoder, self).__init__()

        # self.fc = nn.Linear(latent_dim, hidden_dims[0] * deep_image_size**2)

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.deep_image_size = deep_image_size
        self.out_features = out_features
        self.norm = norm
        self.norm_type = norm_type
        self.start_with_mlp = start_with_mlp

        if activation == 'leakyrelu':
            activation_fn = nn.LeakyReLU
        elif activation == 'relu':
            activation_fn = nn.ReLU
        

        assert latent_dim % deep_image_size**2 == 0, "latent_dim must be divisible by deep_image_size**2"
        
        self.layers = nn.ModuleList()

        if self.start_with_mlp is True:
            self.fc = nn.Linear(latent_dim, hidden_dims[0] * deep_image_size**2)
            in_channels = hidden_dims[0]
        else:
            in_channels = latent_dim // deep_image_size**2

        # decoder
        for i in range(len(hidden_dims)):
            
            out_channels = hidden_dims[i]
            
            conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
            self.layers.append(conv)
            # norm = nn.LayerNorm([out_channels, deep_image_size**(i+2), deep_image_size**(i+2)])
            if self.norm is True:
                if self.norm_type == 'batch':
                    norm = nn.BatchNorm2d(out_channels)
                elif self.norm_type == 'layer':
                    norm = nn.LayerNorm([out_channels, deep_image_size**(i+2), deep_image_size**(i+2)])
                else:
                    raise ValueError("norm_type must be 'batch' or 'layer'")
                self.layers.append(norm)

            activation = activation_fn()

            # set in_channels for next loop. 
            in_channels = out_channels
        
        # finaly layer
        final_layer = nn.Sequential(
            nn.Conv2d(
                hidden_dims[-1], 
                out_channels= self.out_features,
                kernel_size= 3, 
                padding=1
            ),
            nn.Tanh()
        )
        self.layers.append(final_layer)

        self.decoder = nn.Sequential(*self.layers)
    
    def forward(self, x):
        # reshape x into a 2D tensor

        if self.start_with_mlp is True:
            x = self.fc(x)
            x = x.view(-1, self.hidden_dims[0], self.deep_image_size, self.deep_image_size)
        
        if len(x.shape) in (1,2):
            x = x.view(-1, self.latent_dim // self.deep_image_size**2, self.deep_image_size, self.deep_image_size)
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 4:
            pass
        else:
            raise ValueError("x must be a 1D, 2D, 3D, or 4D tensor")

        return self.decoder(x)

class UniqueConsecutive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=0, return_inverse=True):
        unique, indices = torch.unique_consecutive(input, dim=dim, return_inverse=return_inverse)
        ctx.save_for_backward(indices)
        return unique, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices=None):
        indices, = ctx.saved_tensors
        # Count the occurrences of each unique row
        counts = torch.bincount(indices)
        # Expand grad_output according to counts
        expanded_grad = grad_output.repeat_interleave(counts, dim=0)

        return expanded_grad, None, None

unique_consecutive = UniqueConsecutive.apply

class TriplanarDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_objects=1,
        conv_hidden_dims=[512, 512, 512, 512, 512],
        conv_deep_image_size=2,
        sdf_latent_size=128,
        sdf_hidden_dims=[512, 512, 512],
        sdf_weight_norm=True,
        sdf_final_activation='tanh',
        sdf_activation='relu',
        sdf_dropout_prob=0.,
        sum_sdf_features=True,
        
        padding=0.1
    ):
        super(TriplanarDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_objects = n_objects
        self.conv_hidden_dims = conv_hidden_dims
        self.conv_deep_image_size = conv_deep_image_size
        self.sdf_latent_size = sdf_latent_size
        self.sdf_hidden_dims = sdf_hidden_dims
        self.sdf_weight_norm = sdf_weight_norm
        self.sdf_final_activation = sdf_final_activation
        self.sdf_activation = sdf_activation
        self.sdf_dropout_prob = sdf_dropout_prob
        self.sum_sdf_features = sum_sdf_features
        self.padding = padding

        if self.sum_sdf_features is False:
            assert self.sdf_latent_size % 3 == 0, "sdf_latent_size must be divisible by 3 if sum_sdf_features is True"
            vae_out_features = self.sdf_latent_size
        elif self.sum_sdf_features is True:
            vae_out_features = self.sdf_latent_size * 3
        
        self.vae_decoder = VAEDecoder(
            latent_dim=latent_dim,
            out_features=vae_out_features,
            hidden_dims=conv_hidden_dims,
            deep_image_size=conv_deep_image_size,
        )

        self.sdf_decoder = Decoder(
            latent_size=self.sdf_latent_size,
            dims=self.sdf_hidden_dims,
            n_objects=self.n_objects,
            dropout=None if self.sdf_dropout_prob == 0 else list(range(len(self.sdf_hidden_dims))),
            dropout_prob=self.sdf_dropout_prob,
            weight_norm=self.sdf_weight_norm,
            activation=self.sdf_activation,  # "relu" or "sin"
            final_activation=self.sdf_final_activation, #"sin", "linear"
            layer_split=None,
        )
    
    def forward_with_plane_features(self, plane_features, query):
        """
        
        args:
            plane_features: (N, 3 * sdf_latent_size, H, W)
            query: (N, 3)
        """
        latent_size = self.sdf_latent_size
        feat_xz = plane_features[:latent_size, ...]
        feat_yz = plane_features[latent_size:latent_size*2, ...]
        feat_xy = plane_features[latent_size*2:, ...]

        plane_feats_list = []
        plane_feats_list.append(self.sample_plane_features(query, feat_xz, 'xz'))
        plane_feats_list.append(self.sample_plane_features(query, feat_yz, 'yz'))
        plane_feats_list.append(self.sample_plane_features(query, feat_xy, 'xy'))

        if self.sum_sdf_features is True:
            plane_feats = 0
            # sum plane features for each point
            for plane_feat in plane_feats_list:
                plane_feats += plane_feat
            
        elif self.sum_sdf_features is False:
            plane_feats = torch.cat(plane_feats_list, dim=1)
        
        return plane_feats


    def sample_plane_features(self, query, plane_feature, plane):
        """
        args:
            query: (N, 3)
            plane_feature: (sdf_latent_size, H, W)
            plane: 'xz', 'yz', 'xy'
        
        return:
            sampled_feats: (N, sdf_latent_size)
        """
        # normalize coords to [-1, 1] & return 
        grid = self.normalize_coordinates(query.clone(), plane=plane)

        sampled_feats = grid_sample(
            input=plane_feature.unsqueeze(0),
            grid=grid,
            padding_mode='border',
            align_corners=True,
            mode='bilinear'
        ).squeeze(-1).squeeze(0)


        return sampled_feats.T

    def normalize_coordinates(self, query, plane, padding=0.1):
        if plane == 'xy':
            xy = query[:, [0, 1]]
        elif plane == 'xz':
            xy = query[:, [0, 2]]
        elif plane == 'yz':
            xy = query[:, [1, 2]]
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")

        xy_new = xy / (1 + self.padding + 10e-6)
        if xy_new.min() < -1:
            xy_new[xy_new < -1] = -1
        if xy_new.max() > 1:
            xy_new[xy_new > 1] = 1
        
        return xy_new[None, :, None, :]

    def forward(self, x, epoch=None):
        xyz = x[:, -3:]
        latent = x[:, :-3]

        # get unique latent codes
        unique_latent, unique_indices = unique_consecutive(latent, 0, True)
        # get plane features for each unique latent code
        plane_features = self.vae_decoder(unique_latent)
        
        # ALTERNATE METHOD: 
        #    - pre-applocate array of zeros and fill it while
        #    iterating over the data. 

        point_latents = []
        # # # apply forward for data of each unique latent code
        for idx in range(unique_latent.shape[0]):
            # get plane features for each point
            point_latents_ = self.forward_with_plane_features(
                plane_features[idx, :, :, :], 
                xyz[unique_indices == idx, :]
            )
            point_latents.append(point_latents_)
        
        point_latents = torch.cat(point_latents, dim=0)
        sdf_features = torch.cat([point_latents, xyz], dim=1)
        sdf = self.sdf_decoder(sdf_features)

        return sdf


    
    # def forward_with_plane_features(self, plane_features, unique_indices, query):
    #     """
        
    #     args:
    #         plane_features: (N, 3 * sdf_latent_size, H, W)
    #         query: (N, 3)
    #     """
    #     feat_xz = plane_features[:, :self.sdf_latent_size, :, :]
    #     feat_yz = plane_features[:, self.sdf_latent_size:2*self.sdf_latent_size, :, :]
    #     feat_xy = plane_features[:, 2*self.sdf_latent_size:, :, :]

    #     plane_feats_list = []
    #     plane_feats_list.append(self.sample_plane_features(query, unique_indices, feat_xz, 'xz'))
    #     plane_feats_list.append(self.sample_plane_features(query, unique_indices, feat_yz, 'yz'))
    #     plane_feats_list.append(self.sample_plane_features(query, unique_indices, feat_xy, 'xy'))

    #     if self.sum_sdf_features is True:
    #         plane_feats = 0
    #         # sum plane features for each point
    #         for plane_feat in plane_feats_list:
    #             plane_feats += plane_feat
            
    #     elif self.sum_sdf_features is False:
    #         plane_feats = torch.cat(plane_feats_list, dim=1)
        
    #     return plane_feats


    # def sample_plane_features(self, query, unique_indices, plane_feature, plane):
    #     """
    #     args:
    #         query: (N, 3)
    #         unique_indices: (N,)
    #         plane_feature: (N_, sdf_latent_size, H, W)
    #         plane: 'xz', 'yz', 'xy'
        
    #     return:
    #         sampled_feats: (N, sdf_latent_size)
    #     """
    #     # normalize coords to [-1, 1] & return 
    #     grid = self.normalize_coordinates(query.clone(), plane=plane)

    #     # preallocate array for resulting features
    #     # (N, sdf_latent_size)
    #     sampled_feats = torch.zeros(
    #         query.shape[0],
    #         plane_feature.shape[1],
    #         dtype=plane_feature.dtype,
    #         device=plane_feature.device
    #     )

    #     # iterate over plane_features (N_)
    #     # need to do this way because might not have 
    #     # constant number of points per plane
    #     for i in range(plane_feature.shape[0]):
    #         # get single plane feature (1, sdf_latent_size, H, W)
    #         plane_feature_ = plane_feature[i:i+1, :, :, :] 
    #         # get grid of points in shape [1, n_pts, 1, 2]
    #         # n_pts is number of points for this plane (unique_indices==i)
    #         grid_ = grid[None, unique_indices==i, None, :]
            
    #         sample = grid_sample(
    #             input=plane_feature_,
    #             grid=grid_,
    #             padding_mode='border',
    #             align_corners=True,
    #             mode='bilinear'
    #         )
    #         # sample is (1, sdf_latent_size, n_pts, 1)
    #         # need to reshape to (n_pts, sdf_latent_size)
    #         sample = sample[0, :, :, 0].T
    #         sampled_feats[unique_indices==i, :] = sample

    #     return sampled_feats

    # def normalize_coordinates(self, query, plane, padding=0.1):
    #     if plane == 'xy':
    #         xy = query[:, [0, 1]]
    #     elif plane == 'xz':
    #         xy = query[:, [0, 2]]
    #     elif plane == 'yz':
    #         xy = query[:, [1, 2]]
    #     else:
    #         raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    #     xy_new = xy / (1 + self.padding + 10e-6)
    #     if xy_new.min() < -1:
    #         xy_new[xy_new < -1] = -1
    #     if xy_new.max() > 1:
    #         xy_new[xy_new > 1] = 1
        
    #     return xy_new

    # def forward(self, x, epoch=None):
    #     xyz = x[:, -3:]
    #     latent = x[:, :-3]

    #     # get unique latent codes
    #     unique_latent, unique_indices = unique_consecutive(latent, 0, True)
    #     # get plane features for each unique latent code
    #     plane_features = self.vae_decoder(unique_latent)
    #     # plane_features_ = self.vae_decoder(unique_latent)
    #     # # unpack plane features using unique indices
    #     # plane_features = torch.zeros(
    #     #     latent.shape[0],
    #     #     plane_features_.shape[1],
    #     #     plane_features_.shape[2],
    #     #     plane_features_.shape[3],
    #     #     device=latent.device
    #     # )
    #     # for idx in range(unique_latent.shape[0]):
    #     #     plane_features[unique_indices == idx, :] = plane_features_[idx, :]
        
    #     # # get plane features for each point
    #     point_latents = self.forward_with_plane_features(plane_features, unique_indices, xyz)

    #     # get sdf(s) from MLP decoder
    #     sdf_features = torch.cat([point_latents, xyz], dim=1)
    #     sdf = self.sdf_decoder(sdf_features)

    #     return sdf







