import numpy as np
import torch
import torch.nn as nn

from config.default import Config

default = Config()


class OscillatorBank(nn.Module):
    def __init__(self, conf=default):
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

    def forward(self, x):
        harmonics, harm_amps = self.prepare_harmonics(x['f0'], x['c'], 0.)
        phases = self.generate_phases(harmonics)
        signal = self.generate_signal(harm_amps, x['a'], phases)

        return signal

    def prepare_harmonics(self, f0, harm_amps, harm_stretch):
        harmonics = self.harmonics ** (1. + harm_stretch)
        # Hz (cycles per second)
        harmonics = harmonics.unsqueeze(0).unsqueeze(0).repeat(
            f0.shape[0],
            f0.shape[1],
            1) * f0
        # zero out above nyquist
        mask = harmonics > self.sample_rate // 2
        harm_amps = harm_amps.masked_fill(mask, 0.)
        harmonics *= 2 * np.pi  # radians per second
        harmonics /= self.sample_rate  # radians per sample
        harmonics = harmonics.repeat_interleave(self.hop_size, 1)
        return harmonics, harm_amps

    @staticmethod
    def generate_phases(harmonics):
        phases = torch.cumsum(harmonics, dim=1)
        phases %= 2 * np.pi
        return phases

    def generate_signal(self, harm_amps, loudness, phases):
        loudness = loudness.repeat_interleave(self.hop_size, 1)
        harm_amps = harm_amps.repeat_interleave(self.hop_size, 1)
        signal = loudness * harm_amps * torch.sin(phases)
        signal = torch.sum(signal, dim=2) / self.n_harmonics
        return signal

    # TODO: fixing training broke live
    def live(self,
             f0: torch.Tensor,
             loudness: torch.Tensor,
             harm_amps: torch.Tensor,
             harm_stretch: torch.Tensor):
        f0 = f0.unsqueeze(0).unsqueeze(0)
        loudness = loudness.unsqueeze(0).unsqueeze(0)
        harm_amps = harm_amps.unsqueeze(0).unsqueeze(0)
        harm_stretch = harm_stretch.unsqueeze(0).unsqueeze(0)

        harmonics = self.prepare_harmonics(f0, harm_amps, harm_stretch)
        harmonics[0, 0, :] += self.last_phases  # phase offset from last sample
        phases = self.generate_phases(harmonics)
        self.last_phases.data = phases[0, -1, :]  # update phase offset
        signal = self.generate_signal(harm_amps, loudness, phases)

        return signal.squeeze(0).squeeze(0)
