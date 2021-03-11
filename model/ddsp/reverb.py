import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from model.ddsp.filtered_noise import fft_convolve


class Reverb(nn.Module):
    def __init__(self, conf, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = conf.sample_rate
        self.sampling_rate = conf.sample_rate

        self.noise = nn.Parameter((torch.rand(self.length) * 2 - 1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1)
        self.t = nn.Parameter(t, requires_grad=False)

        self.buffer = nn.Parameter(torch.zeros(1, self.length), requires_grad=False)

    def build_impulse(self):
        t = torch.exp(-F.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = F.pad(impulse, (0, lenx - self.length))

        x = fft_convolve(x, impulse)

        return x

    def live_forward(self, x):
        lenx = x.shape[1]
        out = torch.zeros_like(self.buffer)
        out[:, :-lenx] = self.buffer[:, lenx:]
        out[:, -lenx:] = x
        self.buffer[...] = out
        impulse = self.build_impulse()
        x = fft_convolve(out, impulse)

        return x[:, -lenx:]
