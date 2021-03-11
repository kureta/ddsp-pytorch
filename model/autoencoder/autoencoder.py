import torch
from torch import nn
from torch.nn import functional as F  # noqa

from config.default import Config
from model.autoencoder.encoder import Encoder
from train.train import Decoder

default = Config()


class AutoEncoder(nn.Module):
    def __init__(self, conf=default):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.padding = conf.n_fft - conf.hop_length
        self.hop_legth = conf.hop_length

    def forward(self, x):
        x = F.pad(x, (self.padding // 2, self.padding - self.padding // 2))
        z = self.encoder(x)
        signal = self.decoder(z)

        return signal

    # TODO: Every module should be responsible for keeping track of their own
    #       internal state in a `forward_live` method
    def forward_live(self, x, hidden):
        audio_in = torch.from_numpy(x).unsqueeze(0).cuda()
        # We are dropping those samples here
        z = self.encoder(audio_in[:, self.hop_legth // 2:-(self.hop_legth - self.hop_legth // 2)])
        audio_hat, hidden = self.decoder.forward_live(z, hidden)

        return audio_hat, hidden
