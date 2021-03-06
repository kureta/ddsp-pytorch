import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class OscillatorBank(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.n_harmonics = conf.n_harmonics
        self.sample_rate = conf.sample_rate
        self.hop_size = conf.hop_length

        self.harmonics = nn.Parameter(
            torch.arange(1, self.n_harmonics + 1, step=1),
            requires_grad=False
        )
        self.last_phases = nn.Parameter(
            torch.zeros_like(self.harmonics),
            requires_grad=False
        )

    def prepare_harmonics(self, f0, harm_amps):
        # Hz (cycles per second)
        harmonics = self.harmonics.unsqueeze(0).unsqueeze(0).repeat(
            f0.shape[0],
            f0.shape[1],
            1) * f0
        # zero out above nyquist
        mask = harmonics > self.sample_rate // 2
        harm_amps = harm_amps.masked_fill(mask, 0.)
        harm_amps /= harm_amps.sum(-1, keepdim=True)
        harmonics *= 2 * np.pi  # radians per second
        harmonics /= self.sample_rate  # radians per sample
        harmonics = self.rescale(harmonics)
        return harmonics, harm_amps

    @staticmethod
    def generate_phases(harmonics):
        phases = torch.cumsum(harmonics, dim=1)
        phases %= 2 * np.pi
        return phases

    def generate_signal(self, harm_amps, loudness, phases):
        loudness = self.rescale(loudness)
        harm_amps = self.rescale(harm_amps)
        signal = loudness * harm_amps * torch.sin(phases)
        signal = torch.sum(signal, dim=2)
        return signal

    def rescale(self, x):
        return F.interpolate(x.permute(0, 2, 1),
                             scale_factor=self.hop_size,
                             mode='linear').permute(0, 2, 1)

    def forward(self, x):
        harmonics, harm_amps = self.prepare_harmonics(x['f0'], x['c'])
        phases = self.generate_phases(harmonics)
        signal = self.generate_signal(harm_amps, x['a'], phases)

        return signal

    def live(self, x):
        f0 = x['f0']
        harm_amps = x['c']
        loudness = x['a']

        harmonics, harm_amps = self.prepare_harmonics(f0, harm_amps)
        harmonics[0, 0, :] += self.last_phases  # phase offset from last sample
        phases = self.generate_phases(harmonics)
        self.last_phases.data = phases[0, -1, :]  # update phase offset
        signal = self.generate_signal(harm_amps, loudness, phases)

        return signal
