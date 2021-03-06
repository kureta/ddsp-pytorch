import os

import librosa
import numpy as np
import torch
from torch import nn
import torchaudio
from torch.nn import functional as F  # noqa

from crepe import crepe


class F0Encoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.hop_length = conf.hop_length
        self.window_size = conf.n_fft

        self.rs = torchaudio.transforms.Resample(conf.sample_rate, 16000)
        self.min_cents = self.cents_map(0)
        self.max_cents = self.cents_map(359)

        # initialize crepe
        self.model = crepe.Crepe(conf.crepe_capacity)

        # Load weights
        file = os.path.join(os.path.dirname(crepe.__file__),
                            'pretrained',
                            f'{conf.crepe_capacity}.pth')
        self.model.load_state_dict(torch.load(file))

        # This section is commented out because we can continue to train CREPE
        # using analysis by synthesis.
        # Eval mode
        self.model.eval()
        for name, val in self.model.named_parameters():
            val.requires_grad = False

    @staticmethod
    def cents_map(bins):
        return bins * 20 + 1997.3794084376191

    def normalize_cents(self, cents):
        return (cents - self.min_cents) / (self.max_cents - self.min_cents)

    @staticmethod
    def freq_map(cents):
        return 10 * 2 ** (cents / 1200)

    def forward(self, batch):
        with torch.no_grad():
            # example audio duration at the original sample rate (in samples)
            orig_len = batch.shape[1]

            # resample all examples
            x = self.rs(batch)

            # normalize for CREPE
            x -= x.mean(dim=1, keepdim=True)
            x /= x.std(dim=1, keepdim=True)

            # resampled audio duration
            resampled_len = x.shape[1]

            # calculate required hop length at the new sample rate
            resampled_hop_length = int(
                self.hop_length * ((resampled_len - 1024) / (orig_len - self.window_size))
            )

            # get overlapping windows using the new hop length with a window size of 1024
            # as expected by CREPE
            x = x.unfold(1, 1024, resampled_hop_length)

            # save the original batch and time dimensions
            old_shape = x.shape[:2]

            # CREPE is not time aware so smoosh all into a single batch
            x = x.reshape(-1, 1024)

            # CREPE output
            probabilities = self.model(x)

            # Restore time dimension
            probabilities = probabilities.reshape((*old_shape, probabilities.shape[-1]))

            # calculate predicted frequency, harmonicity and normalized pitch
            freq, harmonicity, normalized_cents = self.pitch_argmax(probabilities)

            return freq, harmonicity, probabilities, normalized_cents

    def pitch_weighted(self, probabilities):
        center = probabilities.argmax(dim=-1, keepdim=True)
        return self.pitch_centered(center, probabilities)

    def pitch_centered(self, center, probabilities):
        selection = torch.ones(
            (*center.shape[:2], 9), dtype=torch.int64).to(probabilities.device)

        for idx in range(-4, 5):
            selection[:, :, idx] = center[:, :, 0] + idx + 4

        padded_probs = F.pad(probabilities, (4, 4))
        mask = torch.zeros_like(padded_probs, dtype=torch.bool)
        mask.scatter_(2, selection, True)

        values = torch.masked_select(padded_probs, mask).reshape(
            (*padded_probs.shape[:2], -1))
        cents = self.cents_map(selection - 4)
        product_sum = torch.sum(values * cents, dim=-1, keepdim=True)
        weight_sum = torch.sum(values, dim=-1, keepdim=True)

        cents = product_sum / weight_sum
        freq = self.freq_map(cents)

        harmonicity = probabilities.gather(-1, center)
        normalized_cents = self.normalize_cents(cents)

        return freq, harmonicity, normalized_cents

    def pitch_argmax(self, probabilities):
        bins = probabilities.argmax(dim=-1, keepdim=True)
        cents = self.cents_map(bins)
        freq = self.freq_map(cents)

        harmonicity = probabilities.gather(-1, bins)
        normalized_cents = bins / 359.

        return freq, harmonicity, normalized_cents


class LoudnessEncoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.n_fft = conf.n_fft
        self.hop_length = conf.hop_length
        self.sample_rate = conf.sample_rate
        freqs = np.linspace(0, float(self.sample_rate) / 2, int(1 + self.n_fft // 2), endpoint=True, dtype='float32')
        a_weight = librosa.A_weighting(freqs)
        self.a_weight = nn.Parameter(torch.from_numpy(a_weight), requires_grad=False)

    def forward(self, signal):
        stft = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            return_complex=True,
        ).permute(0, 2, 1)
        stft = torch.log10(torch.abs(stft) + 1e-20) * 20
        stft += self.a_weight
        # TODO: arbitrary noise floor of -90db, and we assume max loudness = 0db
        stft = stft / 90 + 1

        loudness = torch.mean(stft, dim=-1, keepdim=True)

        return loudness


class Encoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.f0_encoder = F0Encoder(conf)
        self.loudness_encoder = LoudnessEncoder(conf)

    def forward(self, x):
        result = {}
        f0, harmonicity, probabilities, normalized_cents = self.f0_encoder(x)
        loudness = self.loudness_encoder(x)
        result['f0'] = f0
        result['harmonicity'] = harmonicity
        result['loudness'] = loudness
        result['probabilities'] = probabilities
        result['normalized_cents'] = normalized_cents

        return result
