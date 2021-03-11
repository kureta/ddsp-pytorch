import torch
from torch import nn
from torch.nn import functional as F  # noqa

from model.autoencoder.decoder import Decoder
from model.autoencoder.encoder import Encoder
from model.ddsp.filtered_noise import FilteredNoise
from model.ddsp.harmonic_oscillator import OscillatorBank
from model.ddsp.reverb import Reverb


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.ddsp = OscillatorBank()
        self.noise = FilteredNoise()
        self.reverb = Reverb()

    def forward(self, x):
        # TODO: fix magic numbers
        x = F.pad(x, (256 + 512, 256 + 512))
        z = self.encoder(x)
        ctrl = self.decoder(z)
        harmonics = self.ddsp(ctrl)
        noise = self.noise(ctrl)

        signal = harmonics + noise
        signal = self.reverb(signal)

        return signal

    # TODO: Every module should be responsible for keeping track of their own
    #       internal state in a `forward_live` method
    def forward_live(self, x, hidden):
        audio_in = torch.from_numpy(x).unsqueeze(0).cuda()
        # We are dropping those samples here
        z = self.encoder(audio_in[:, 256:-256])
        ctrl, hidden = self.decoder(z, hidden)
        harmonics = self.ddsp.live(ctrl)
        noise = self.noise(ctrl)

        audio_hat = harmonics + noise
        audio_hat = self.reverb.live_forward(audio_hat)

        return audio_hat.cpu().squeeze(0).numpy(), hidden
