import torch
from torch import nn
from torch.nn import functional as F  # noqa

from model.autoencoder.encoder import Encoder
from train.train import DDSPDecoder


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.ddsp = DDSPDecoder()

    def forward(self, x):
        # TODO: fix magic numbers
        x = F.pad(x, (256 + 512, 256 + 512))
        z = self.encoder(x)
        signal = self.ddsp(z)

        return signal

    # TODO: Every module should be responsible for keeping track of their own
    #       internal state in a `forward_live` method
    def forward_live(self, x, hidden):
        audio_in = torch.from_numpy(x).unsqueeze(0).cuda()
        # We are dropping those samples here
        z = self.encoder(audio_in[:, 256:-256])
        audio_hat, hidden = self.ddsp.forward_live(z, hidden)

        return audio_hat, hidden
