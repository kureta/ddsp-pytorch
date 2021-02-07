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
        harmonics = self.prepare_harmonics(f0, harm_amps, harm_stretch)
        phases = self.generate_phases(harmonics)
        signal = self.generate_signal(harm_amps, loudness, phases)

        return signal

    def prepare_harmonics(self, f0, harm_amps, harm_stretch):
        harmonics = self.harmonics ** (1. + harm_stretch)
        harmonics *= f0  # Hz (cycles per second)
        # zero out above nyquist
        harm_amps[harmonics > self.sample_rate // 2] = 0.
        harmonics *= 2 * np.pi  # radians per second
        harmonics /= self.sample_rate  # radians per sample
        harmonics = harmonics.repeat_interleave(self.hop_size, 1)
        return harmonics

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
