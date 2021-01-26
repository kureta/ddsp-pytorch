"""
2020_01_29 - 2020_02_03
Loudness Extractor / Envelope Follower
TODO :
    check appropriate gain structure
    GPU test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class LoudnessExtractor(nn.Module):
    def __init__(self,
                 sr=16000,
                 attenuate_gain=2.,
                 device='cuda'):
        super(LoudnessExtractor, self).__init__()

        self.sr = sr
        self.n_fft = 2048

        self.device = device

        self.attenuate_gain = attenuate_gain
        self.smoothing_window = nn.Parameter(torch.hann_window(self.n_fft, dtype=torch.float32),
                                             requires_grad=False).to(self.device)

    def torch_a_weighting(self, frequencies, min_db=-45.0):
        """
        Compute A-weighting weights in Decibel scale (codes from librosa) and 
        transform into amplitude domain (with DB-SPL equation).
        
        Argument: 
            FREQUENCIES : tensor of frequencies to return amplitude weight
            min_db : mininum decibel weight. appropriate min_db value is important, as 
                exp/log calculation might raise numeric error with float32 type. 
        
        Returns:
            weights : tensor of amplitude attenuation weights corresponding to the FREQUENCIES tensor.
        """

        # Calculate A-weighting in Decibel scale.
        frequency_squared = frequencies ** 2
        const = torch.tensor([12200, 20.6, 107.7, 737.9]) ** 2.0
        weights_in_db = 2.0 + 20.0 * (torch.log10(const[0]) + 4 * torch.log10(frequencies)
                                      - torch.log10(frequency_squared + const[0])
                                      - torch.log10(frequency_squared + const[1])
                                      - 0.5 * torch.log10(frequency_squared + const[2])
                                      - 0.5 * torch.log10(frequency_squared + const[3]))

        # Set minimum Decibel weight.
        if min_db is not None:
            weights_in_db = torch.max(weights_in_db, torch.tensor([min_db], dtype=torch.float32).to(self.device))

        # Transform Decibel scale weight to amplitude scale weight.
        weights = torch.exp(torch.log(torch.tensor([10.], dtype=torch.float32).to(self.device)) * weights_in_db / 10)

        return weights

    def forward(self, input_signal):
        """
        Compute A-weighted Loudness Extraction
        Input:
            z['audio'] : batch of time-domain signals
        Output:
            output_signal : batch of reverberated signals
        """
        sliced_windowed_signal = input_signal * self.smoothing_window

        # TODO: something is fishy.
        sliced_signal = torch.rfft(sliced_windowed_signal, 1, onesided=False)

        sliced_signal_loudness_spectrum = sliced_signal[:, :, 0] ** 2 + sliced_signal[:, :, 1] ** 2

        freq_bin_size = self.sr / self.n_fft
        frequencies = torch.tensor([(freq_bin_size * i) % (0.5 * self.sr) for i in range(self.n_fft)]).to(self.device)
        a_weights = self.torch_a_weighting(frequencies)

        a_weighted_sliced_signal_loudness_spectrum = sliced_signal_loudness_spectrum * a_weights
        a_weighted_sliced_signal_loudness = torch.sqrt(
            torch.sum(a_weighted_sliced_signal_loudness_spectrum, 1)) / self.n_fft * self.attenuate_gain

        return a_weighted_sliced_signal_loudness
