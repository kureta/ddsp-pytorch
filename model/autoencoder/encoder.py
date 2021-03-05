import os

import torch
import torch.nn as nn
import torchaudio
from crepe import crepe

from config.default import Config

default = Config()


class F0Encoder(nn.Module):
    def __init__(self, conf=default):
        super().__init__()
        self.hop_length = conf.hop_length
        self.window_size = conf.n_fft

        self.rs = torchaudio.transforms.Resample(conf.sample_rate, 16000)

        self.freq_map = nn.Parameter(torch.linspace(0, 7180, 360) + 1997.3794084376191, requires_grad=False)
        # initialize crepe
        self.model = crepe.Crepe(conf.crepe_capacity)

        # Load weights
        file = os.path.join(os.path.dirname(crepe.__file__), 'pretrained', f'{conf.crepe_capacity}.pth')
        self.model.load_state_dict(torch.load(file))

        # Eval mode
        self.model.eval()
        for name, val in self.model.named_parameters():
            val.requires_grad = False

    def forward(self, batch):
        with torch.no_grad():
            orig_len = batch.shape[1]
            x = self.rs(batch)
            x -= x.mean(dim=1, keepdims=True)
            x /= x.std(dim=1, keepdims=True)
            new_len = x.shape[1]
            hop_length = int(self.hop_length * ((new_len - 1024) / (orig_len - self.window_size)))
            x = x.unfold(1, 1024, hop_length)
            old_shape = x.shape[:2]
            x = x.reshape(-1, 1024)
            probabilities = self.model(x)
            bins = probabilities.argmax(dim=1)
            scaled_bins = bins / 360.
            cents = self.freq_map[bins]
            freq = 10 * 2 ** (cents / 1200)
            harmonicity = probabilities.gather(1, bins.unsqueeze(1))
            freq = freq.reshape((*old_shape, 1))
            harmonicity = harmonicity.reshape((*old_shape, 1))
            probabilities = probabilities.reshape((*old_shape, 360))
            scaled_bins = scaled_bins.reshape((*old_shape, 1))

            return freq, harmonicity, probabilities, scaled_bins


class LoudnessEncoder(nn.Module):
    def __init__(self, conf=default):
        super().__init__()
        self.conf = conf

    def forward(self, x):
        x = x.unfold(1, self.conf.n_fft, self.conf.hop_length)
        # rms = torch.sqrt(torch.sum(x ** 2, dim=2) / self.conf.n_fft)
        dbfs = 20 * torch.log10(torch.sum(torch.abs(x), dim=2) / self.conf.n_fft + 1e-10)
        return (dbfs + 98) / 98


class Encoder(nn.Module):
    def __init__(self, conf=default):
        super().__init__()
        self.conf = conf

        self.f0_encoder = F0Encoder()
        self.loudness_encoder = LoudnessEncoder()

    def forward(self, x):
        result = {}
        f0, harmonicity, probabilities, scaled_bins = self.f0_encoder(x)
        loudness = self.loudness_encoder(x).unsqueeze(-1)
        result['f0'] = f0
        result['harmonicity'] = harmonicity
        result['loudness'] = loudness
        result['probabilities'] = probabilities
        result['scaled_bins'] = scaled_bins

        return result
