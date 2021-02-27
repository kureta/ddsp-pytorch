import torch
import torch.nn as nn
import torchaudio

from config.default import Config
from model.autoencoder.decoder import MLP

default = Config()


# TODO: keep track of GRU hidden states for live
class FeatureEncoder(nn.Module):
    def __init__(self, n_features=1, conf=default):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=conf.sample_rate,
            n_mfcc=conf.n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=conf.n_fft,
                hop_length=conf.hop_length,
                n_mels=conf.n_mels,
                f_min=conf.f_min,
                f_max=conf.f_max
            ),
        )

        # self.norm = nn.InstanceNorm1d(conf.n_mfcc, affine=True)

        self.gru = nn.GRU(
            input_size=conf.n_mfcc,
            hidden_size=conf.encoder_gru_units,
            num_layers=1,
            batch_first=True,
        )
        # self.mlp = MLP(conf.n_mfcc, 512, 3)
        self.dense = nn.Linear(conf.encoder_gru_units, n_features)

    def forward(self, x):
        x = self.mfcc(x)
        # x = self.norm(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        # x = self.mlp(x)
        x = self.dense(x)
        return x


class Encoder(nn.Module):
    def __init__(self, conf=default):
        super().__init__()
        self.conf = conf

        self.f0_encoder = FeatureEncoder(1)
        self.loudness_encoder = FeatureEncoder(1)
        if conf.use_z:
            self.z = FeatureEncoder(conf.z_units)

    def forward(self, x):
        result = {}
        f0 = self.f0_encoder(x)
        f0 = torch.sigmoid(f0) * 2 + 6
        f0 = torch.pow(2, f0)
        loudness = self.loudness_encoder(x)
        loudness = torch.sigmoid(loudness) + 0.1
        result['f0'] = f0
        result['loudness'] = loudness

        if self.conf.use_z:
            z = self.z(x)
            result['z'] = z

        return result
