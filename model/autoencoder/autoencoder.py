import torch
from torch import nn
from torch.nn import functional as F

from model.autoencoder.decoder import Decoder
from model.autoencoder.encoder import Encoder
from model.ddsp.filtered_noise import FilteredNoise
from model.ddsp.harmonic_oscillator import OscillatorBank


# TODO: Implement the following:
#       * LoudnessExtractor (that takes frequencies into account)
#       * Reverb (also real-time)

# TODO: There is a problem in either the AutoEncoder or the OscillatorBank
#       Frames have "seams" inbetween


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.ddsp = OscillatorBank()
        self.noise = FilteredNoise()

    def forward(self, x):
        # TODO: fix magic numbers
        x = F.pad(x, (256 + 512, 256 + 512))
        z = self.encoder(x)
        ctrl = self.decoder(z)
        harmonics = self.ddsp(ctrl)
        noise = self.noise(ctrl)

        signal = harmonics + noise

        return signal

    # TODO: similarly to the decoder, we can ad an if statement to forward
    #       and drop this forward live
    def forward_live(self, x, hidden):
        audio_in = torch.from_numpy(x).unsqueeze(0).cuda()
        # We are dropping those samples here
        z = self.encoder(audio_in[:, 256:-256])
        ctrl, hidden = self.decoder(z, hidden)
        harmonics = self.ddsp.live(ctrl)
        noise = self.noise(ctrl)

        audio_hat = harmonics + noise

        return audio_hat.cpu().squeeze(0).numpy(), hidden
