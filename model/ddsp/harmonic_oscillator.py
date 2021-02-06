import numpy as np
import torch
import torch.nn as nn


class OscillatorBank(nn.Module):
    def __init__(self, n_harmonics: int = 100, sample_rate: int = 44100, hop_size: int = 64):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_size = hop_size

        self.harmonics = nn.Parameter(torch.arange(1, self.n_harmonics + 1, step=1), requires_grad=False)
        self.last_phases = nn.Parameter(torch.zeros_like(self.harmonics), requires_grad=False)

    def forward(self,
                f0: torch.Tensor,
                loudness: torch.Tensor,
                harm_amps: torch.Tensor,
                harm_stretch: torch.Tensor):
        harmonics = self.harmonics ** (1. + harm_stretch)
        harmonics *= f0  # Hz (cycles per second)
        # zero out above nyquist
        harm_amps[harmonics > self.sample_rate // 2] = 0.

        harmonics *= 2 * np.pi  # radians per second
        harmonics /= self.sample_rate  # radians per sample

        harmonics = harmonics.repeat(self.hop_size, 1)
        harmonics[0, :] += self.last_phases  # phase offset from last sample
        phases = torch.cumsum(harmonics, dim=0)
        phases %= 2 * np.pi

        self.last_phases.data = phases[-1, :]

        signal = loudness * harm_amps * torch.sin(phases)
        signal = torch.sum(signal, dim=1) / self.n_harmonics

        return signal

    def live(self,
             f0: torch.Tensor,
             loudness: torch.Tensor,
             harm_amps: torch.Tensor,
             harm_stretch: torch.Tensor):
        f0 = f0.unsqueeze(0).unsqueeze(0)
        loudness = loudness.unsqueeze(0).unsqueeze(0)
        harm_amps = harm_amps.unsqueeze(0).unsqueeze(0)
        harm_stretch = harm_stretch.unsqueeze(0).unsqueeze(0)

        return self.forward(f0, loudness, harm_amps, harm_stretch)
