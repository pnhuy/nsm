from torch import nn
import diffusion_net
import torch
import sys
sys.path.append('/bmrNAS/people/aagatti/programming/PyTorch-VAE/')
from torch.nn import functional as F
from models.types_ import *
from .diffusion_layers import DiffusionNetEncoder
from GenerativeAnatomy.utils import assd
try:
    from pytorch3d.loss import chamfer_distance
    _chamfer_ = True
except:
    print('Error importing `chamfer_distance` from pytorch3d.loss')
    _chamfer_ = False

activations = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'linear': nn.Linear,
            'softmax': nn.Softmax,
            'leakyrelu': nn.LeakyReLU
        }

functional_activations = {
    'elu': nn.functional.elu,
    'relu': nn.functional.relu,
    'tanh': nn.functional.tanh,
    'sigmoid': nn.functional.sigmoid,
    'linear': nn.functional.linear,
    'softmax': nn.functional.softmax,
    'leakyrelu': nn.functional.leaky_relu
}


class GeneralVAE(nn.Module):

    # FUNCTIONS THAT SHOULD LIKELY BE CUSTOM CREATED FOR
    # EACH NEW VAE MODEL ARE INCLUDED IN THE BELOW
    def __init__(self,
                 n_points: int,
                 n_mesh_dims: int = 3,
                 C_width: int = 256,
                 C_in: int = None, # THIS WILL BE COMPUTED BASED ON INPUTS - UNLESS SPECIFIED, NEEDS TO BE SPECIFIED FOR DIFFUSION_NET
                 latent_dim: int = 64,
                 dropout: float=0.1,
                 decode_hidden_dims: list = None,
                 verbose: bool = False,
                 kld_weight: float = 0.00025,
                 # CHANGE BELOW TO BE SPECIFIC DECODER TYPE & THEN IT IS USEFUL FOR ALL ENCODER TYPES
                 decoder_type = 'mlp', #CHANGED FROM diffusion_decoder to decoder_type
                 conv_decode_connector_dense: bool=True,
                 decoder_diffnet_n_blocks: int=5,
                 decoder_variance: bool=False,
                 decoder_final_layer: str='dense',
                 mean_shape = None,
                 diff_decoder_verts=None,
                 diff_decoder_faces=None,
                 diff_decoder_k_eig: int=128,
                 diff_decoder_op_cache_dir=None,
                 device='cuda:0',
                 activation='relu',
                 recon_loss_fx='mse',
                 penalize_predicting_mean=True,
                 mean_penalty_weight=0.1,
                 rand_pts_assd=False,
                 n_pts_assd=100,
                 variational=True,
                 predict_residuals=False,
                 **kwargs) -> None:
        super(GeneralVAE, self).__init__()
        
        self.n_points = n_points
        self.n_mesh_dims = n_mesh_dims
        self.C_width = C_width
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.decode_hidden_dims = decode_hidden_dims
        self.verbose = verbose
        self.kld_weight = kld_weight
        self.decoder_type = decoder_type
        self.conv_decode_connector_dense = conv_decode_connector_dense
        self.decoder_diffnet_n_blocks = decoder_diffnet_n_blocks
        self.decoder_variance = decoder_variance
        self.decoder_final_layer = decoder_final_layer
        self.mean_shape = mean_shape
        self.diff_decoder_verts = diff_decoder_verts
        self.diff_decoder_faces = diff_decoder_faces
        self.diff_decoder_k_eig = diff_decoder_k_eig
        self.diff_decoder_op_cache_dir = diff_decoder_op_cache_dir
        # self.mean_faces = mean_faces
        self.device = device
        self.activation = activation
        self.recon_loss_fx = recon_loss_fx
        self.rand_pts_assd = rand_pts_assd
        self.n_pts_assd = n_pts_assd
        self.variational = variational
        self.predict_residuals = predict_residuals
        self.diff_decoder_idx = None
        self.penalize_predicting_mean = penalize_predicting_mean
        self.mean_penalty_weight = mean_penalty_weight

        if self.decoder_type == 'diffusion_net':
            if self.diff_decoder_verts is None:
                if self.mean_shape is not None:
                    self.diff_decoder_verts = self.mean_shape
                else:
                    raise Exception('No verts locations for diffusion decoder')
            if self.diff_decoder_faces is None:
                raise Exception('No faces provided for duffusion decoder - LOOK INTO POINT CLOUD VERSION OF DECODER')
            
            # Get properties of mean mesh for diffusion net
            if self.verbose is True:
                print('diff_decoder_verts shape: ', self.diff_decoder_verts.shape)
                print('diff_decoder_faces shape: ', self.diff_decoder_faces.shape)
            self.frames, self.mass, self.L, self.evals, self.evecs, self.gradX, self.gradY = diffusion_net.geometry.get_operators(
                self.diff_decoder_verts, 
                self.diff_decoder_faces, 
                k_eig=self.diff_decoder_k_eig, 
                op_cache_dir=self.diff_decoder_op_cache_dir
            )
            # assign properties to the device (gpu) 
            # also expand first dim (batch)
            self.mass = self.mass.to(self.device).unsqueeze(0)
            self.L = self.L.to(self.device).unsqueeze(0)
            self.evals = self.evals.to(self.device).unsqueeze(0)
            self.evecs = self.evecs.to(self.device).unsqueeze(0)
            self.gradX = self.gradX.to(self.device).unsqueeze(0)
            self.gradY = self.gradY.to(self.device).unsqueeze(0)
            if self.verbose is True:
                print('mass shape', self.mass.shape)
                print('L shape', self.L.shape)
                print('evals shape', self.evals.shape)
                print('evecs shape', self.evecs.shape)
                print('gradX shape', self.gradX.shape)
                print('gradY shape', self.gradY.shape)
            self.diff_decoder_faces.unsqueeze(0)

        if C_in is None:
            self.C_in = self.n_points * self.n_mesh_dims
        else:
            self.C_in = C_in
        
        if mean_shape is not None:
            self.mean_shape = self.mean_shape.to(self.device)
        
        if self.verbose is True:
            print('decoder_type', decoder_type)
            print('kwargs', kwargs)
        
        self.init_decoder()
        


    def init_encoder(self):
        """
        SETUP CODE FOR INITIALIZING ENCODER IN HERE 
        """
        return
    
    def init_variational_sampling(self, in_size):
        '''
            SAMPLE LATENT SPACE
        '''
        self.fc_mu = nn.Linear(in_size, self.latent_dim)
        self.add_module(f"encode_fc_mu", self.fc_mu)
        self.fc_var = nn.Linear(in_size, self.latent_dim)
        self.add_module(f"encode_fc_var", self.fc_var)
    
    def init_decoder(self):
        """
        SETUP CODE FOR INITIALIZING DECODER IN HERE
        """
        '''
            BUILD DECODER
        '''
        self.decoder_modules = []
        if self.decoder_type in ('mlp', 'pointnet'):
            if (self.conv_decode_connector_dense is True) & (self.decoder_type == 'pointnet'):
                # linear layer
                self.decoder_modules.append(nn.Linear(1, self.latent_dim))
                self.add_module(f'decoder_linear_layer_hidden_to_conv', self.decoder_modules[-1])

                # Activation
                self.decoder_modules.append(
                    activations[self.activation]()
                )
                self.add_module(f"decode_activation_hidden_to_conv", self.decoder_modules[-1])

                # dropout
                self.decoder_modules.append(
                    nn.Dropout(p=self.dropout)
                )
                self.add_module(f"decode_dropout_hidden_to_conv", self.decoder_modules[-1])
            
            for layer_idx, hidden_size in enumerate(self.decode_hidden_dims):
                if layer_idx == 0:
                    if self.decoder_type == 'pointnet':
                        # POINTNET DECODER IS CURRENTLY "WRONG"
                        # IT EFFECTIVELY JUST USES LOTS OF RAM TO GET TO 
                        # SOMETHING THAT COULD BE MUCH SIMLER. 
                        # IT APPLIED A 1D CONV TO AN ARRAY OF: 
                        # (batch_size, hidden_size, n_points)
                        # WHERE BATCH HIDDEN_SIZE IS THE SAME VALUES
                        # FOR EVERYTHING n_points FOR A GIVEN 
                        # PARTICIPANT
                        #
                        # CORRECTION
                        # POINTNET DECODER IS FINE, BUT MIGHT NOT BE OPTIMAL.
                        # CURRENTLY CREATES (n_batch, 1, hidden_size) and then applies
                        # 1d conv - which applies the same MLP for each node in hidden_size
                        # to the data in the second axis/dimension. This might not be
                        # expressive enough, but it is not replicating the exact same thing
                        # over and over again like I was worried about. 
                        # MIGHT BE USEFUL TO ADD A DENSE LAYER OPTION THAT TURNS
                        # (n_batch, 1, hidden_size) into (n_batch, hidden_size, hidden_size)
                        if self.conv_decode_connector_dense is False:
                            in_size=1
                        elif self.conv_decode_connector_dense is True:
                            in_size = self.latent_dim
                    elif self.decoder_type == 'mlp':
                        in_size = self.latent_dim
                else:
                    in_size = self.decode_hidden_dims[layer_idx-1]
                if self.decoder_type == 'mlp':
                    self.decoder_modules.append(
                        nn.Linear(in_size, hidden_size)
                    )
                    self.add_module(f"decode_linear_{layer_idx}", self.decoder_modules[-1])
                elif self.decoder_type == 'pointnet':
                    self.decoder_modules.append(
                        nn.Conv1d(in_channels=in_size, out_channels=hidden_size, kernel_size=1) #input_channels, output_chan
                    )
                    self.add_module(f"decode_conv_{layer_idx}", self.decoder_modules[-1])
                # Activation
                self.decoder_modules.append(
                    activations[self.activation]()
                )
                self.add_module(f"decode_activation_{layer_idx}", self.decoder_modules[-1])
                # dropout
                self.decoder_modules.append(
                    nn.Dropout(p=self.dropout)
                )
                self.add_module(f"decode_dropout_{layer_idx}", self.decoder_modules[-1])
            
            layer_idx += 1
            
            if (self.decoder_type == 'pointnet') & (hidden_size != self.n_points):
                # setup so the number of channels = the number of nodes
                # will max pool after this to go from: (batch_size, 10, n_nodes) => (batch_size, n_nodes)
                # can then reshape to be (batch_size, n_nodes, 1) and apply another convolution (or dense layer) to get 
                # a mesh sized output (batch_Size, n_nodes, n_mesh_dims)
                self.decoder_modules.append(
                    nn.Conv1d(in_channels=hidden_size, out_channels=self.n_points, kernel_size=1) #input_channels, output_chan
                )
                self.add_module(f"decode_conv_{layer_idx}", self.decoder_modules[-1])

                # Activation
                self.decoder_modules.append(
                    activations[self.activation]()
                )
                self.add_module(f"decode_activation_{layer_idx}", self.decoder_modules[-1])
                # dropout
                self.decoder_modules.append(
                    nn.Dropout(p=self.dropout)
                )
                self.add_module(f"decode_dropout_{layer_idx}", self.decoder_modules[-1])

                layer_idx += 1

            # ADD A LAYER THAT LINEARLY PREDICTS THE OUTPUT (no activation)
            # THIS IS TO GET THE ACTUAL PREDICTED OUTPUT/SHAPE
            if (self.decoder_type == 'mlp'):
                self.decoder_modules.append(
                    nn.Linear(self.decode_hidden_dims[-1], self.n_mesh_dims*self.n_points)
                )
                self.add_module(f"decode_linear_{layer_idx}", self.decoder_modules[-1])
            elif (self.decoder_type == 'pointnet'):
                if self.decoder_final_layer == 'conv':
                    # convolve to end up with the expected size. 
                    self.decoder_modules.append(
                        nn.Conv1d(in_channels=1, out_channels=self.n_mesh_dims, kernel_size=1) #input_channels, output_chan
                    )
                    self.add_module(f"decode_conv_final_{layer_idx}", self.decoder_modules[-1])
                elif self.decoder_final_layer == 'dense':
                    self.decoder_modules.append(
                        nn.Linear(in_features=1, out_features=self.n_mesh_dims) #input_channels, output_chan
                    )
                    self.add_module(f"decode_linear_final_{layer_idx}", self.decoder_modules[-1])

        elif self.decoder_type == 'diffusion_net':
            if type(self.decode_hidden_dims) in (list, tuple):
                if len(self.decode_hidden_dims) > 1:
                    raise Exception(f'only one set of hidden dims used/setup for diff net decoder! {self.decode_hidden_dims} provided')
                else:
                    self.decode_hidden_dims = self.decode_hidden_dims[0]
            
            # LEAVING THE BELOW IN CASE WE WANT to GET MULTI LAYER DIFFUSION NET IN THE FUTURE
            # if type(self.C_width) in (list, tuple):
            #     self.encoder_module = DiffusionNetEncoder(
            #         C_in=self.C_in,
            #         C_out=self.C_out,
            #         C_width=self.C_width, 
            #         last_activation=None,
            #         outputs_at=self.encoder_output_at, 
            #         dropout=self.dropout
            #     )

            # Add an option in future to have a linear layer: 
            # would first need to expand dims (batch, hidden, 1)
            # then linear to get (batch, hidden, n_points)
            # then permute so that its (batch, n_points, hidden)
            # then this would be the input into the diffusion layer
            layer_idx = 0
            diff_layer_tracker = 0
            if self.conv_decode_connector_dense is True:
                # linear layer
                self.decoder_modules.append(nn.Linear(self.latent_dim, self.n_points * self.latent_dim))
                self.add_module(f'decoder_linear_layer_{layer_idx}', self.decoder_modules[-1])
                diff_layer_tracker += 1

                # Activation
                self.decoder_modules.append(
                    activations[self.activation]()
                )
                self.add_module(f"decode_activation_{layer_idx}", self.decoder_modules[-1])
                diff_layer_tracker += 1

                # dropout
                self.decoder_modules.append(
                    nn.Dropout(p=self.dropout)
                )
                self.add_module(f"decode_dropout_{layer_idx}", self.decoder_modules[-1])
                diff_layer_tracker += 1
                layer_idx += 1


            # This is the same regardless of linear layer between or not. 
            self.decoder_modules.append(diffusion_net.layers.DiffusionNet(
                C_in=self.latent_dim,
                C_out=self.n_mesh_dims,
                C_width=self.decode_hidden_dims, 
                N_block=self.decoder_diffnet_n_blocks, 
                last_activation=None,  #NO ACTIVATION BECAUSE TRYING TO OUTPUT X/Y/Z LOCATIONS
                outputs_at='vertices', 
                dropout=self.dropout)
            )

            self.add_module(f'diffusion_decoder_{layer_idx}', self.decoder_modules[-1])
            self.diff_decoder_idx = diff_layer_tracker
            
        else:
            raise NotImplementedError(f'Only mlp, pointnet, diffusion_net decoders are implemented, {self.decoder_type} was requested')
        
        if self.decoder_variance is True:
            # THIS VARIANCE DECODER IS DENSE RIGHT NOW - COULD REPLACE AND MAKE IT ADAPTABLE IN THE FUTURE IF WE WANTED. 
            if type(self.decode_hidden_dims) in (int, float):
                self.decode_hidden_dims = [self.decode_hidden_dims,]
            self.decoder_variance_modules = []
            for layer_idx, hidden_size in enumerate(self.decode_hidden_dims):
                if layer_idx == 0:
                    in_size = self.latent_dim
                else:
                    in_size = self.decode_hidden_dims[layer_idx-1]
                self.decoder_variance_modules.append(
                    nn.Linear(in_size, hidden_size)
                )
                self.add_module(f"decode_variance_linear_{layer_idx}", self.decoder_variance_modules[-1])
                # Activation
                self.decoder_variance_modules.append(
                    activations[self.activation]()
                )
                self.add_module(f"decode_variance_activation_{layer_idx}", self.decoder_variance_modules[-1])
                # dropout
                self.decoder_variance_modules.append(
                    nn.Dropout(p=self.dropout)
                )
                self.add_module(f"decode_variance_dropout_{layer_idx}", self.decoder_variance_modules[-1])
            
            layer_idx += 1
            
            # ADD A LAYER THAT LINEARLY PREDICTS THE OUTPUT (softplus activation to reduce the scale of the variance relative to
            # the actual shape)
            # THIS IS TO GET THE ACTUAL PREDICTED OUTPUT/SHAPE
            self.decoder_variance_modules.append(
                nn.Linear(self.decode_hidden_dims[-1], self.n_points * self.n_mesh_dims)  #this needs to be shape (n_batch, n_pts * dims_per_point) so that it can be re-shaped to be (n_batch, n_points, n_mesh_dims)
            )
            self.add_module(f"decode_variance_linear_{layer_idx}", self.decoder_variance_modules[-1])
            # Activation
            self.decoder_variance_modules.append(
                nn.Softplus()
            )
            self.add_module(f"decode_variance_activation_{layer_idx}", self.decoder_variance_modules[-1])
            # dropout

        return
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x D]
        """
        if len(z.size()) == 1:
            z = torch.unsqueeze(z, 0)
        
        if self.decoder_variance is True:
            z_var = z
        
        if len(self.mean_shape.shape) == 2:
            self.mean_shape = self.mean_shape.unsqueeze(0)

        if len(z.size()) == 2:
            if (self.decoder_type in ('diffusion_net', 'pointnet')):
                if self.conv_decode_connector_dense is True:
                    # if using dense layer before diffusion net, then we want to apply the first
                    # dense layer to keep the number of hidden layers & to upsample the number of points (in the
                    # last axis) like: [n_batch, n_hidden, 1] after applying the dense layer we will permute 
                    # have shape: [n_batch, n_hidden, n_points] and we will permute these back to be:
                    # [n_batch, n_points, n_hidden]
                    if self.verbose is True:
                        print('Leaving latent space the same (no reshape to connect hidden to diffnet)')
                    # z = z[:, :, None]
                    # UPDATED - NO LONGER RESHAPING IF USING DENSE TO CONNECT LATENT TO DIFFNET
                    # IF RESHAPE, THEN WE HAVE INPUT VECTOR = a single value. The f(x) applied to 
                    # it then gets applied to every other value in the mesh. Therefore, every single
                    # node on the diffusionnet get the exact same inputs => this definitely seems
                    # suboptimal. 
                elif self.conv_decode_connector_dense is False:
                    # if using pointnet autoencoder or the no dense layer with diffusion net we want
                    # the unitary dimension to be 1. 
                    # for pointnet: 
                    #     We want to retain our number of hidden dims in the final axis/dimension becuase
                    #     if instead we make it unitary then this will just be like a dense later applied to
                    #     the outputs of the hidden layer. Insted we want the convolution layet to apply 
                    #     to each of the hidden layer values separately. Otherwise, doing max pool at the end
                    #     doesnt make sense. 
                    #     THIS DOESNT MAKE SENSE BECAUSE WE HAVE THE SAME VALUES AT EACH NODE. THUS THE CONV1D
                    #     IS REDUNDANT!
                    # for diffusion_net:
                    #     We want to repeat the hidden dims (Which will be in the final dim/axis) for each of
                    #     The nodes in the surface mesh (second dimension/axis). I am not certain if this will work
                    #     becuase in theory each node is getting the exact same data/information. However, the 
                    #     different nodes should have different gradient information....   
                    if self.verbose is True:
                        print('reshaping point net input')
                    z = z[:, None, :]

                    if self.decoder_type == 'diffusion_net':
                        # apply the repeating function so that we have the right input size for the diffnet layer. 
                        z = z.repeat(1, self.n_points, 1)

        for idx, layer in enumerate(self.decoder_modules):
            if self.verbose is True:
                print('decoder layer: ', idx)
            
            if (self.decoder_type == 'diffusion_net') & (idx == self.diff_decoder_idx):
                # to provide all of the extra inputs to diffusionnet predictions 
                z = layer(z, mass=self.mass, L=self.L, evals=self.evals, evecs=self.evecs, gradX=self.gradX, gradY=self.gradY, faces=self.diff_decoder_faces)
            else:
                # apply layer normally with no extra wizardry
                z = layer(z)

            if (idx == len(self.decoder_modules)-2) & (self.decoder_type == 'pointnet'):
                if self.verbose is True:
                    print('performing pointnet maxpool')
                z = torch.max(z, 2)[0]
                if self.decoder_final_layer == 'conv':
                    # this is because the convolution is applied over the second layer - so we want it to be singleton
                    z = z[:, None, :]
                elif self.decoder_final_layer == 'dense':
                    # this is because the dense later is applied over the last layer - so we want it to be singleton
                    z = z[:, :, None]
                    
            elif (self.decoder_type == 'diffusion_net'):
                # after applying dense layer, need to permute the last 2 dimensions
                # so that the input to diffusion_net has shape: (n_batch, n_pts, hidden_size)
                if (idx == 0) & (self.conv_decode_connector_dense is True):
                    # z = z.permute(0, 2, 1) 
                    # ABOVE WAS WHEN APPLYING DENSE TO INPUT SHAPED: 
                    # (n_batch, n_hidden_dims, 1) => (n_batch, n_hidden_dims, n_pts)
                    # we permuted to get the n_pts in second dim. Now, need to reshape
                    # instead. 
                    z = z.view(-1, self.n_points, self.latent_dim)

        if self.verbose:
            print('z shape: ', z.shape)
            print('n_points: ', self.n_points)
            print('n_mesh_dims:', self.n_mesh_dims)
        # result = torch.reshape(z, (self.n_points, self.n_mesh_dims))
        # reformat the shape of the data to keep the same number of batches (-1)
        # and have the points be in their original shape (n_points, mesh_dims)
        if (self.decoder_type == 'pointnet') & (self.decoder_final_layer == 'conv'):
            # undoing the permutation needed to apply convolutions over the second axis/dimension
            # this is not needed for dense because it is already applied over the appropriate dimension(s). 
            z = z.permute(0, 2, 1)
        elif self.decoder_type == 'mlp':
            z = z.view(-1, self.n_points, self.n_mesh_dims)

        result = z

        if self.verbose is True:
            print('result shape', result.shape)
            print('latter dims', (self.n_points, self.n_mesh_dims))
        
        assert result.shape[1] == self.n_points
        assert result.shape[2] == self.n_mesh_dims

        if self.predict_residuals is True:
            if self.verbose is True:
                print('self.mean_shape', self.mean_shape.shape)
            result = result + self.mean_shape
        
        if self.decoder_variance is True:
            for idx, layer in enumerate(self.decoder_variance_modules):
                if self.verbose is True:
                    print('decoder layer: ', idx)
                z_var = layer(z_var)
            
            z_var = z_var.view(-1, self.n_points, self.n_mesh_dims)

            if self.verbose is True:
                print('z_var shape: ', z_var.shape)
            
            assert z_var.shape[1] == self.n_points
            assert z_var.shape[2] == self.n_mesh_dims
            assert len(z_var.shape) == 3

            # Add the variance values to the mean values to get the prediction. 
            result += z_var

            if self.verbose is True:
                print('result shape before output', result.shape)
            
            return result, z_var
        else:
            if self.verbose is True:
                print('result shape before output', result.shape)

            return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        # SD = sqrt(variance)
        # because we ahve a logvar, this is a linear
        # function. So, taking exp(0.5*logvar) is the
        # faster way to calculate SD from logvar
        std = torch.exp(0.5 * logvar)
        # then we add random values from a normal
        # distribution N(0, 1)
        eps = torch.randn_like(std)
        #add sd * rand to give noise to each data point 
        return eps * std + mu 

    """
    Below are Loss Function Related 
    """
    def r2_loss(self, output, target, verbose=False, **kwargs):
        if self.mean_shape is None:
            target_mean = torch.mean(target, dim=0)
        elif self.mean_shape is not None:
            # if len(self.mean_shape.shape) < 3:
            #     print('unsqueezing the first dimension, current shape is: ',self.mean_shape.shape)
            #     self.mean_shape = self.mean_shape.unsqueeze(0)
            #     print('self.mean_shape after unsqueeze is: ', self.mean_shape.shape)
            # if self.mean_shape.shape[0] != output.shape[0]:
            #     print('repeating first dimension to match batch size - current shape is: ', self.mean_shape.shape)
            #     self.mean_shape = self.mean_shape.repeat(output.shape[0], 1, 1)
            #     print('self.mean_shape after repeat is: ', self.mean_shape.shape)

            target_mean = self.mean_shape
        
        if verbose is True:
            print(f'Target shape: {target.shape}')
            print(f'Target mean shape: {target_mean.shape}')

        if self.recon_loss_fx in ('mse', 'l2'):
            ss_tot = torch.sum((target - target_mean) ** 2)
            ss_res = torch.sum((target - output) ** 2)
        elif self.recon_loss_fx in ('mae', 'l1'):
            ss_tot = F.l1_loss(target, target_mean, reduction='sum')
            ss_res = F.l1_loss(target, output, reduction='sum')
        elif self.recon_loss_fx == 'assd':
            assd_mean_shape = assd(target, target_mean, rand_pts=self.rand_pts_assd, n_pts=self.n_pts_assd)
            ss_tot = assd_mean_shape
            ss_res = self.recons_loss
        elif self.recon_loss_fx == 'assd_mse':
            mse_mean_shape = F.mse_loss(target, target_mean)
            assd_mean_shape = assd(target, target_mean, rand_pts=self.rand_pts_assd, n_pts=self.n_pts_assd)
            recons_loss = (kwargs['assd_proportion'] * assd_mean_shape) + (1-kwargs['assd_proportion']) * mse_mean_shape
            ss_tot = recons_loss
            ss_res = self.recons_loss
        elif self.recon_loss_fx == 'chamfer':
            if _chamfer_ is False:
                raise Exception('Could not import chamfer loss!')
            if target_mean.shape[0] != target.shape[0]:
                print(f'repeating `target_mean` {target.shape[0]} times to match batch size')
                target_mean = target_mean.repeat(target.shape[0], 1, 1)
            chamfer_mean_shape, _ = chamfer_distance(target, target_mean)
            ss_tot = chamfer_mean_shape
            ss_res = self.recons_loss
            if self.verbose is True:
                print('ss_total / chamfer_mean_shape: ', chamfer_mean_shape)
        else:
            raise Exception('No loss function specified!')

        r2 = 1 - ss_res / ss_tot
        return r2

    # @staticmethod
    def loss_function(self,
                    *args,
                    **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        if self.verbose is True:
            print('length of args: ', len(args))
            for i in range(len(args)):
                print(f'\tn items in args # {i}: ', len(args[i]))
            
        if self.decoder_variance is True:
            recons = args[0][0]
            var = args[0][1]
            input_ = args[0][2]
            mu = args[0][3]
            log_var = args[0][4]
        else:
            recons = args[0][0]
            input_ = args[0][1]
            mu = args[0][2]
            log_var = args[0][3]

        if len(log_var.shape) == 1:
            log_var = torch.unsqueeze(log_var, 0)
        if len(mu.shape) == 1:
            mu = torch.unsqueeze(mu, 0)


        if self.verbose is True:
            print('recons shape', recons.shape)
            print('input_ shape', input_.shape)
            
        # https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/beta_vae.py#L137-L148
    #         kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        if self.recon_loss_fx in ('mse', 'l2'):
            if self.decoder_variance is False:
                recons_loss = F.mse_loss(recons, input_)
            else:
                if self.verbose is True:
                    print('var shape: ', var.shape)
                squared_error = (recons - input_)**2
                point_wise_error = 0.5 * (-var).exp() * squared_error + 0.5 * var
                recons_loss = torch.mean(point_wise_error)
        elif self.recon_loss_fx in ('l1', 'mae'):
            if self.decoder_variance is False:
                recons_loss = F.l1_loss(recons, input_)
            else:
                if self.verbose is True:
                    print('var shape: ', var.shape)
                abs_error = torch.abs(recons - input_)
                point_wise_error = 0.5 * (-var).exp() * abs_error + 0.5 * var
                recons_loss = torch.mean(point_wise_error)
        elif self.recon_loss_fx == 'assd':
            if self.decoder_variance is False:
                recons_loss = assd(recons, input_, rand_pts=self.rand_pts_assd, n_pts=self.n_pts_assd)
                self.recons_loss = recons_loss
            else:
                raise Exception('decoder variance not implemented for ASSD, yet.')
        elif self.recon_loss_fx == 'assd_mse':
            if self.decoder_variance is False:
                mse = F.mse_loss(recons, input_)
                assd_ = assd(recons, input_, rand_pts=self.rand_pts_assd, n_pts=self.n_pts_assd)
                recons_loss = (kwargs['assd_proportion'] * assd_) + (1-kwargs['assd_proportion']) * mse
                self.recons_loss = recons_loss
            else:
                raise Exception('decoder variance not implemented for ASSD/MSE, yet.')
        elif self.recon_loss_fx == 'chamfer':
            if self.decoder_variance is False:
                recons_loss, _ = chamfer_distance(recons, input_)
                self.recons_loss = recons_loss
            else:
                raise Exception('decoder variance not implemented for chamfer, yet.')


        r2 = self.r2_loss(recons, input_, **kwargs)

        if self.verbose is True:
            print('log_var shape', log_var.shape)
            print('mu shape', mu.shape)

        # Calculate KL loss per batch first
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        if self.verbose is True:
            print('kld_loss before averaging batches size is: ', kld_loss.shape)
            print('kld_loss', kld_loss)

        kld_loss = torch.abs(kwargs['C_kld'] - kld_loss)   # the C from `Understanding disentangling in β-VAE` : 

        # Take the average of the KL terms across all of the different batches. 
        # do this before normalization to reduce number of operations
        kld_loss = torch.mean(kld_loss, dim = 0)

        # Normalize kld based on the latent size & the input size
        # This is according to Beta-VAE ICLR 2017
        # kld_loss = (kld_loss * self.latent_dim) / (self.n_points * self.n_mesh_dims) => THIS IS ALREADY BEING DONE BY TAKING THE MEAN? 
        # COULD ALSO NORMALIZE KLD LOSS TO THE MINI BATCH SIZE RELATIVE TO DATASET SIZE. 
        
        if self.verbose is True:
            print('kld_loss shape after averaging batches size is: ', kld_loss.shape)
            print('kld_loss', kld_loss)

        if self.variational is True:
            loss = recons_loss + kwargs['kld_weight'] * kld_loss
        else:
            loss = recons_loss
        
        if self.penalize_predicting_mean is True:
            if self.recon_loss_fx == 'mse':
                mean_penalty = 1 / (F.mse_loss(recons, self.mean_shape))
            elif self.recon_loss_fx == 'assd':
                mean_penalty = 1 / assd(recons, self.mean_shape, rand_pts=self.rand_pts_assd, n_pts=self.n_pts_assd)
            loss += self.mean_penalty_weight * mean_penalty

        if self.verbose is True:
            print('recons_loss', recons_loss)
            print('kld_weight', self.kld_weight)
            print('kld_loss', kld_loss)
            print('loss', loss)

        return {
            'loss': loss, 
            'Reconstruction_Loss':recons_loss.detach(), 
            'KLD':kld_loss.detach(),
            'R2': r2.detach()}

    # SAMPLE RANDOMLY FROM LATENT SPACE TO CREATE NEW OUTPUTS
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
    
    def forward(self, x_in) -> List[Tensor]:
        if len(x_in.size()) == 2:
            x_in = torch.unsqueeze(x_in, 0)
        mu, log_var = self.encode(x_in)

        if self.variational is False:
            z = mu 
        else:
            z = self.reparameterize(mu, log_var)
        
        if self.decoder_variance is True:
            x_hat, x_var = self.decode(z)

            if self.verbose is True:
                print('x_hat shape', x_hat.shape)
                print('x_var shape', x_var.shape)
                print('x_in shape', x_in.shape)
                print('mu shape', mu.shape)
                print('log_var shape', log_var.shape)

            return [x_hat, x_var, x_in, mu, log_var]
        else:
            return  [self.decode(z), x_in, mu, log_var]

    '''
    BELOW ARE FUNCTIONS THAT SHOULD PROBABLY BE RE-WRITTEN FOR EACH NEW SUBCLASS
    '''

    def encode(self, x_in) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x D]
        :return: (Tensor) List of latent codes
        """
        
        # Run steps necessary for encoding the input(s)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        if self.verbose:
            print('mu shape: ', mu.shape)

        # Return mu & log_var
        return [mu, log_var]    

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class DiffNetVAE(GeneralVAE):
    def __init__(self,
                # DIFFUSIONNET SPECIFIC
                N_blocks: int = 5,
                C_out: int = 128, # CAN MAYBE REPLACE THIS FOR JUST USING LATENT_DIM IN THE FUTURE
                 #CAN MAYBE REPLACE THIS WITH JUST
                encoder_output_at='global_mean',
                **kwargs) -> None:
        super().__init__(**kwargs)
            
        self.N_blocks = N_blocks
        self.C_out = C_out
        self.encoder_output_at = encoder_output_at
        
        # initialize stuff needed to build the encoder
        self.init_encoder()
        
        # initialize stuff needed to sample from the latent space 
        self.init_variational_sampling(self.C_out)


    def init_encoder(self):
        '''
            Build Encoder
        '''
        if self.verbose is True:
            print('n_mesh_dims ', self.n_mesh_dims)
            print('c_width ', self.C_width)
            print('N_block ', self.N_blocks)
        if type(self.C_width) in (list, tuple):
            raise Exception('this works - but currently unclear if int or list/tuple specified. If want to use this function, uncomment/fix this bug')
            # self.encoder_module = DiffusionNetEncoder(
            #     C_in=self.C_in,
            #     C_out=self.C_out,
            #     C_width=self.C_width, 
            #     last_activation=None,  # SHOULD THIS HAVE AN ACTIVATION THE SAME AS THE REGULAR DIFFNET ENCODER BELOW? 
            #     outputs_at=self.encoder_output_at, 
            #     dropout=self.dropout
            # )
        elif type(self.C_width) is int:
            self.encoder_module = diffusion_net.layers.DiffusionNet(
                                                C_in=self.C_in,
                                                C_out=self.C_out,
                                                C_width=self.C_width, 
                                                N_block=self.N_blocks, 
                                                last_activation=functional_activations[self.activation],
                                                outputs_at=self.encoder_output_at, 
                                                dropout=self.dropout)

        self.add_module('diffusion_encoder', self.encoder_module)


    def encode(self, x_in, mass, L, evals, evecs, gradX, gradY, faces) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x D]
        :return: (Tensor) List of latent codes
        """
        x = self.encoder_module(x_in, mass, L, evals, evecs, gradX, gradY, faces=faces)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        if self.verbose:
            print('mu shape: ', mu.shape)

        return [mu, log_var]

    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY, faces) -> List[Tensor]:
        mu, log_var = self.encode(x_in, mass, L, evals, evecs, gradX, gradY, faces)
        if self.variational is False:
            z = mu 
        else:
            z = self.reparameterize(mu, log_var)
        
        if self.decoder_variance is True:
            x_hat, x_var = self.decode(z)

            if self.verbose is True:
                print('x_hat shape', x_hat.shape)
                print('x_var shape', x_var.shape)
                print('x_in shape', x_in.shape)
                print('mu shape', mu.shape)
                print('log_var shape', log_var.shape)

            return [x_hat, x_var, x_in, mu, log_var]
        else:
            return  [self.decode(z), x_in, mu, log_var]

class VanillaVAE(GeneralVAE):
    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        
        # initialize stuff needed to build the encoder
        self.init_encoder()
        
        # initialize stuff needed to sample from the latent space 
        self.init_variational_sampling(self.C_width[-1])


    def init_encoder(self):
        '''
            Build Encoder
        '''
        self.encoder_modules = []
        for layer_idx, layer_size in enumerate(self.C_width):
            if layer_idx == 0:
                in_size = self.C_in
            else:
                in_size = self.C_width[layer_idx-1]
            self.encoder_modules.append(
                nn.Linear(in_size, layer_size)
            )
            self.add_module(f'encoder_{layer_idx}', self.encoder_modules[-1])
            # Activation
            self.encoder_modules.append(
                activations[self.activation]()
            )
            self.add_module(f"encode_activation_{layer_idx}", self.encoder_modules[-1])
            # dropout
            self.encoder_modules.append(
                nn.Dropout(p=self.dropout)
            )
            self.add_module(f"encoder_dropout_{layer_idx}", self.encoder_modules[-1])


    def encode(self, x_in) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x D]
        :return: (Tensor) List of latent codes
        """
        if len(x_in.size()) == 2:
            x_in = torch.unsqueeze(x_in, 0)

        x = torch.flatten(x_in, start_dim=1)

        if self.verbose is True:
            print('x shape', x.shape)

        for layer_idx, layer in enumerate(self.encoder_modules):
            if self.verbose is True:
                print('layer_idx', layer_idx)
            x = layer(x)
            if self.verbose is True:
                print('x shape after layer', x.shape)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        if self.verbose:
            print('mu shape: ', mu.shape)

        return [mu, log_var]


class PointNetVAE(GeneralVAE):
    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        # initialize stuff needed to build the encoder
        self.init_encoder()
        
        # initialize stuff needed to sample from the latent space 
        self.init_variational_sampling(self.C_width[-1])
    
    def init_encoder(self):
        '''
            Build Encoder
        '''
        if self.verbose is True:
            print('n_points', self.n_points)
            print('mesh_dims ', self.n_mesh_dims)

        self.encoder_modules = []
        for layer_idx, layer_size in enumerate(self.C_width):
            if layer_idx == 0:
                in_size = self.n_mesh_dims
            else:
                in_size = self.C_width[layer_idx-1]
            self.encoder_modules.append(
                nn.Conv1d(in_channels=in_size, out_channels=layer_size, kernel_size=1) #input_channels, output_chan
            )
            self.add_module(f'encoder_{layer_idx}', self.encoder_modules[-1])
            # Activation
            self.encoder_modules.append(
                activations[self.activation]()
            )
            self.add_module(f"encode_activation_{layer_idx}", self.encoder_modules[-1])
            # dropout
            self.encoder_modules.append(
                nn.Dropout(p=self.dropout)
            )
            self.add_module(f"encoder_dropout_{layer_idx}", self.encoder_modules[-1])


    def encode(self, x_in) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x D]
        :return: (Tensor) List of latent codes
        """
        if len(x_in.size()) == 2:
            x_in = torch.unsqueeze(x_in, 0)

        x = x_in.permute(0, 2, 1) # NEED TO HAVE CHANNELS (x/y/z) IN THE SECOND DIMENSION
        
        if self.verbose is True:
            print('x shape', x.shape)

        for layer_idx, layer in enumerate(self.encoder_modules):
            if self.verbose is True:
                print('layer_idx', layer_idx)
            x = layer(x)
            if self.verbose is True:
                print('x shape after layer', x.shape)
        
        # MAX POOLING: 
        x = torch.max(x, 2)[0]
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        if self.verbose:
            print('mu shape: ', mu.shape)

        return [mu, log_var]

"""
DiffusionNetPlus was a DiffNet + dense decoder that was used as an extension of diffusionnet
It is commented out because it has not been fully updated to the current methods of inheriting
from GeneralVAE
"""
# class DiffNetPlus(nn.Module):
#     """
#     https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383
#     """
#     def __init__(
#         self,
#         C_in,
#         dense_sizes=(128, 64),
#         C_out=2, 
#         C_width=128,
#         N_block=4, 
#         mlp_hidden_dims=None, 
#         dropout=False, 
#         with_gradient_features=True, 
#         with_gradient_rotations=True, 
#         diffusion_method='spectral',
#         activation='relu',
#         final_activation=None,
#     ):
        
#         super(DiffNetPlus, self).__init__()
#         self.DiffusionNet = diffusion_net.layers.DiffusionNet(
#             C_in=C_in, 
#             C_out=dense_sizes[0], 
#             C_width=C_width,
#             N_block=N_block, 
#             last_activation=None, 
#             outputs_at='global_mean', 
#             mlp_hidden_dims=mlp_hidden_dims, 
#             dropout=dropout, 
#             with_gradient_features=with_gradient_features, 
#             with_gradient_rotations=with_gradient_rotations, 
#             diffusion_method=diffusion_method
#         )

#         if (final_activation is None) or (final_activation == 'log_softmax'):
#             final_activation = lambda x : torch.nn.functional.log_softmax(x,dim=-1)
            
#         self.dense_blocks = []
#         for idx, dense_size in enumerate(dense_sizes):
#             if idx == len(dense_sizes)-1:
#                 self.dense_blocks.append(nn.Linear(in_features=dense_size, out_features=C_out))
#                 self.add_module("block_"+str(idx), self.dense_blocks[-1])
#                 if dropout > 0:
#                     self.dense_blocks.append(nn.Dropout(p=dropout))
                
#             else:
#                 self.dense_blocks.append(nn.Linear(in_features=dense_size, out_features=dense_sizes[idx+1]))
#                 self.add_module("block_"+str(idx), self.dense_blocks[-1])
#             self.dense_blocks.append(activations[activation]())
#         if type(final_activation) is str:
#             self.final_act = activations[final_activation]()
#         else:
#             self.final_act = final_activation
            
                                             
#     def forward(self, x_in, mass, L, evals, evecs, gradX, gradY, faces):
#         x = self.DiffusionNet(x_in, mass, L, evals, evecs, gradX, gradY, faces=faces)
#         for layer in self.dense_blocks:
#             x = layer(x)
#         x = self.final_act(x)
        
#         x.to(torch.float)
        
#         return x

def build_model(config, mean_shape=None, diff_decoder_faces=None):
    c_in_dict = {'xyz':3, 'hks':16}

    print(config)

    if config['model_type'] == 'vae':
        model = DiffNetVAE(
            C_in = c_in_dict[config['input_features']],
            n_points = config['n_points'],
            n_mesh_dims = config['n_mesh_dims'],
            C_out = config['hidden_dims'],
            C_width = config['c_width'],
            N_blocks = config['n_blocks'],
            latent_dim = config['latent_dims'],
            dropout = config['dropout'],
            decode_hidden_dims = config['decode_hidden_dims'],
            activation=config['activation'],
            verbose=config['verbose'],
            kld_weight = config['kld_weight'], #float = 0.00025
            decoder_type = config['decoder_type'],
            diff_decoder_faces=diff_decoder_faces,
            diff_decoder_k_eig=config['k_eig'],
            decoder_variance = config['decoder_variance'],
            decoder_final_layer = config['decoder_final_layer'],
            decoder_diffnet_n_blocks = config['decoder_diffnet_n_blocks'],
            conv_decode_connector_dense = config['conv_decode_connector_dense'],
            mean_shape = mean_shape,
            recon_loss_fx=config['recon_loss_fx'],
            n_pts_assd=config['n_pts_assd'],
            rand_pts_assd=config['rand_pts_assd'],
            penalize_predicting_mean=config['penalize_predicting_mean'],
            mean_penalty_weight=config['mean_penalty_weight'],
            variational=config['variational'],
            predict_residuals=config['predict_residuals'],
        )
    elif config['model_type'] == 'vanilla_vae':
        model = VanillaVAE(
            n_points = config['n_points'],
            n_mesh_dims = config['n_mesh_dims'],
            C_width =  config['c_width'],
            latent_dim = config['latent_dims'],
            dropout = config['dropout'],
            decode_hidden_dims = config['decode_hidden_dims'],
            kld_weight = config['kld_weight'],
            mean_shape = mean_shape,
            device='cuda:0',
            verbose=config['verbose'],
            activation=config['activation'],
            decoder_type = config['decoder_type'],
            diff_decoder_faces=diff_decoder_faces,
            diff_decoder_k_eig=config['k_eig'],
            decoder_variance = config['decoder_variance'],
            decoder_final_layer = config['decoder_final_layer'],
            decoder_diffnet_n_blocks = config['decoder_diffnet_n_blocks'],
            conv_decode_connector_dense = config['conv_decode_connector_dense'],
            recon_loss_fx=config['recon_loss_fx'],
            rand_pts_assd=config['rand_pts_assd'],
            n_pts_assd=config['n_pts_assd'],
            penalize_predicting_mean=config['penalize_predicting_mean'],
            mean_penalty_weight=config['mean_penalty_weight'],
            variational=config['variational'],
            predict_residuals=config['predict_residuals']

        )
    elif config['model_type'] == 'pointnet':
        model = PointNetVAE(
            n_points = config['n_points'],
            n_mesh_dims = config['n_mesh_dims'],
            C_width =  config['c_width'],
            latent_dim = config['latent_dims'],
            dropout = config['dropout'],
            decode_hidden_dims = config['decode_hidden_dims'],
            kld_weight = config['kld_weight'],
            mean_shape = mean_shape,
            device='cuda:0',
            verbose=config['verbose'],
            activation=config['activation'],
            decoder_type = config['decoder_type'],
            diff_decoder_faces=diff_decoder_faces,
            diff_decoder_k_eig=config['k_eig'],
            decoder_variance = config['decoder_variance'],
            decoder_final_layer = config['decoder_final_layer'],
            decoder_diffnet_n_blocks = config['decoder_diffnet_n_blocks'],
            conv_decode_connector_dense = config['conv_decode_connector_dense'],
            recon_loss_fx=config['recon_loss_fx'],
            rand_pts_assd=config['rand_pts_assd'],
            n_pts_assd=config['n_pts_assd'],
            penalize_predicting_mean=config['penalize_predicting_mean'],
            mean_penalty_weight=config['mean_penalty_weight'],
            variational=config['variational'],
            predict_residuals=config['predict_residuals']
        )
    elif config['dense_layers'] is None:
        model = diffusion_net.layers.DiffusionNet(C_in=c_in_dict[config['input_features']],
                                                  C_out=config['n_classes'],
                                                  C_width=config['c_width'], 
                                                  N_block=config['n_blocks'], 
                                                  last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                                  dropout=config['dropout'])
    else:
        model = DiffNetPlus(
            C_in=c_in_dict[config['input_features']],
            dense_sizes=config['dense_layers'],
            C_out=config['n_classes'], 
            C_width=config['c_width'],
            N_block=config['n_blocks'], 
            dropout=config['dropout'], 
            final_activation=config['final_activation']
        )


    return model