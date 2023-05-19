import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

PROGRESSIVE_PARAMS = {
    'n_layers': 3,
    'layers': {
        5: { # Or -3, -2, -1.... 
            'start_epoch': 200,
            'warmup_epochs': 200,
        },
        6:{ # Or -3, -2, -1.... 
            'start_epoch': 600,
            'warmup_epochs': 200,
        },
        7:{ # Or -3, -2, -1.... 
            'start_epoch': 1010,
            'warmup_epochs': 200,
        },
    }
}

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        n_objects=1,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        latent_dropout=False,
        activation="relu",  # "relu" or "sin"
        final_activation="tanh", #"sin", "linear"
        concat_latent_input=False,
        progressive_add_depth=False,
        progressive_depth_params=PROGRESSIVE_PARAMS
    ):
        """
        latent_size (int): size of the latent input vector to the decoder network
        dims (list of ints): list containing the size of each layer in MLP. 
        n_objects (int): number of objects to predict
        dropout (list of ints): where to apply dropout to the encoder
        dropout_prob (float) : probability with which dropout is applied
        norm_layers (list of ints): where to apply weightnorm/batchnorm to the decoder
        latent_in (list of ints): where to repeat the latent vector in the decoder
        weight_norm (bool): whether to apply weight normalization
        xyz_in_all (bool): for deepSDF decoder, include XYZ at each layer
        use_tanh (bool): for deepSDF decoder, tanh final layer to [0, 1]
        latent_dropout (bool): for deepSDF decoder, run dropout on input latent vector
        """
        super(Decoder, self).__init__()

        def make_sequence():
            return []
        
        self._activation_ = activation
        self._final_activation_ = final_activation

        #layers:
        # 0: input
        # 1 to N-hidden: NN
        # -1: output
        if n_objects == 1:
            dims = [latent_size + 3] + dims + [1]
        else:
            dims = [latent_size + 3] + dims + [n_objects]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        self.progressive_add_depth = progressive_add_depth
        self.progressive_depth_params = progressive_depth_params

        # self.activation = activation

        for layer in range(0, self.num_layers - 1):
            if concat_latent_input == False:
                in_dim = dims[layer]
                if (layer + 1 in latent_in):
                    out_dim = dims[layer + 1] - dims[0]
                else:
                    out_dim = dims[layer + 1]
            elif concat_latent_input == True:
                out_dim = dims[layer + 1]
                if layer in latent_in:
                    in_dim = dims[layer] + dims[0]
                else:
                    in_dim = dims[layer]
            else:
                in_dim = dims[layer]
                out_dim = dims[layer + 1]
            if self.xyz_in_all and layer != self.num_layers - 2:
                #TODO: Should this also include logic for
                # concat_latent_input? - right now it assumes not concat
                out_dim -= 3

            lin = nn.Linear(in_dim, out_dim)

            # be sure to specially initialize layer weights 
            # for the sine activation based network (Siren). 
            if activation == 'sin':
                with torch.no_grad():
                    num_input = lin.weight.size(-1)
                    if layer == 0:
                        torch.nn.init.uniform_(lin.weight, -1 / num_input, 1 / num_input)
                    else:
                        torch.nn.init.uniform_(lin.weight, -np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(lin),
                )
            else:
                setattr(self, "lin" + str(layer), lin)

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))
        
        # SETUP ACTIVATION
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sin":
            self.activation = Sine()
        
        # SETUP FINAL ACTIVATION
        if final_activation == "sin":
            self.final_activation = Sine()
        elif final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "linear":
            self.final_activation = None
        else:
            raise ValueError("Invalid final activation")
        

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.epoch = None

    # input: N x (L+3)
    def forward(self, input, epoch=None):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                xi = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                xi = torch.cat([x, xyz], 1)
            else:
                xi = x
            
            if (self.progressive_add_depth is True) and (layer in self.progressive_depth_params['layers']):
                # use this as a way to store the progress of the network so far.
                # this way if we try to use the partly tuned model for inference, it will be able to 
                # use the weights that have been tuned so far.
                if epoch is not None:
                    self.epoch = epoch
                # progresive tuning of latter layers is from Curriculum DeepSDF
                # code was adapted from:
                #https://github.com/haidongz-usc/Curriculum-DeepSDF
                start = self.progressive_depth_params['layers'][layer]['start_epoch']
                warmup = self.progressive_depth_params['layers'][layer]['warmup_epochs']
                end = start + warmup
                if self.epoch < start:
                    # before start... dont apply this block at all.
                    # we are doing continue (vs break) because the 
                    # final layer should still be applied. 
                    continue
                elif start < self.epoch < end:
                    # during warmup... linearly phase this block in
                    # https://github.com/haidongz-usc/Curriculum-DeepSDF/blob/ca216dda8edc6435139a6f657c45800791be94a7/networks/deep_sdf_decoder_train.py#L113
                    new_weight = ((self.epoch - start)/warmup)
                    new_weight = new_weight**2
                    base_weight = 1 - new_weight
                    # base_weight = ((end-self.epoch)/warmup)

                    x_base = xi * base_weight
                    x_new = lin(xi) * new_weight
                    x = x_base + x_new
                else:
                    # after start + warmup epochs just apply this block as a normal layer
                    x = lin(xi)
            else:
                x = lin(xi)
            
            # only apply normalization/ regular activation to 
            # hidden layers (not output)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                    and (self._activation_ != "sin")
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                
                x = self.activation(x)

                if self.dropout is not None and layer in self.dropout:  #and (self._activation_ != "sin")
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x