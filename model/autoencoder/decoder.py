"""
Implementation of decoder network architecture of DDSP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.default import Config


default = Config()


class MLP(nn.Module):
    """
    MLP (Multi-layer Perception). 

    One layer consists of what as below:
        - 1 Dense Layer
        - 1 Layer Norm
        - 1 ReLU

    constructor arguments :
        n_input : dimension of input
        n_units : dimension of hidden unit
        n_layer : depth of MLP (the number of layers)
        relu : relu (default : nn.ReLU, can be changed to nn.LeakyReLU, nn.PReLU for example.)

    input(x): torch.tensor w/ shape(B, ... , n_input)
    output(x): torch.tensor w/ (B, ..., n_units)
    """

    def __init__(self, n_input, n_units, n_layer, relu=nn.ReLU, inplace=False):
        super().__init__()
        self.n_layer = n_layer
        self.n_input = n_input
        self.n_units = n_units
        self.inplace = inplace

        self.add_module(
            f"mlp_layer1",
            nn.Sequential(
                nn.Linear(n_input, n_units),
                nn.LayerNorm(normalized_shape=n_units),
                relu(inplace=self.inplace),
            ),
        )

        for i in range(2, n_layer + 1):
            self.add_module(
                f"mlp_layer{i}",
                nn.Sequential(
                    nn.Linear(n_units, n_units),
                    nn.LayerNorm(normalized_shape=n_units),
                    relu(inplace=self.inplace),
                ),
            )

    def forward(self, x):
        for i in range(1, self.n_layer + 1):
            x = self.__getattr__(f"mlp_layer{i}")(x)
        return x


class Decoder(nn.Module):
    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_freq: 65
        gru_units: 512

    input(dict(f0, z(optional), l)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        *note dimension of z is not specified in the paper.

    output : a dict object which contains key-values below
        f0 : same as input
        c : torch.tensor w/ shape(B, time, n_harmonics) which satisfies sum(c) == 1
        a : torch.tensor w/ shape(B, time) which satisfies a > 0
        H : noise filter in frequency domain. torch.tensor w/ shape(B, frame_num, filter_coeff_length)
    """

    def __init__(self, config=default):
        super().__init__()

        self.config = config

        self.mlp_f0 = MLP(n_input=1, n_units=config.decoder_mlp_units, n_layer=config.decoder_mlp_layers)
        self.mlp_loudness = MLP(n_input=1, n_units=config.decoder_mlp_units, n_layer=config.decoder_mlp_layers)
        if config.use_z:
            self.mlp_z = MLP(
                n_input=config.z_units, n_units=config.decoder_mlp_units, n_layer=config.decoder_mlp_layers
            )
            self.num_mlp = 3
        else:
            self.num_mlp = 2

        self.gru = nn.GRU(
            input_size=self.num_mlp * config.decoder_mlp_units,
            hidden_size=config.decoder_gru_units,
            num_layers=1,
            batch_first=True,
        )

        self.mlp_gru = MLP(
            n_input=config.decoder_gru_units,
            n_units=config.decoder_mlp_units,
            n_layer=config.decoder_mlp_layers,
            inplace=False,
        )

        # one element for overall loudness
        self.dense_harmonic = nn.Linear(config.decoder_mlp_units, config.n_harmonics)
        self.dense_loudness = nn.Linear(config.decoder_mlp_units, 1)
        # self.dense_filter = nn.Linear(config.decoder_mlp_units, config.n_noise_filters)

    def forward(self, batch):
        f0 = batch["f0"]
        loudness = batch["loudness"]

        if self.config.use_z:
            z = batch["z"]
            latent_z = self.mlp_z(z)

        latent_f0 = self.mlp_f0(f0)
        latent_loudness = self.mlp_loudness(loudness)

        if self.config.use_z:
            latent = torch.cat((latent_f0, latent_z, latent_loudness), dim=-1)
        else:
            latent = torch.cat((latent_f0, latent_loudness), dim=-1)

        if 'hidden' in batch:
            latent, h = self.gru(latent, batch['hidden'])
        else:
            latent, h = self.gru(latent)
        latent = self.mlp_gru(latent)

        c = F.softmax(self.dense_harmonic(latent))
        a = Decoder.modified_sigmoid(self.dense_loudness(latent))

        # H = self.dense_filter(latent)
        # H = Decoder.modified_sigmoid(H)

        # return dict(f0=batch["f0"], a=a, c=c, H=H, hidden=h)
        return dict(f0=batch["f0"], a=a, c=c, hidden=h, loudness=batch['loudness'])

    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a
