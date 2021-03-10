import torch
import torch.nn as nn
import torch.fft as fft

from config.default import Config
from model.ddsp.filtered_noise import fft_convolve

default = Config()


class FIRReverb(nn.Module):
    def __init__(self, conf=default, device="cuda"):
        super().__init__()

        # default reverb length is set to 3sec.
        # thus this model can max out t60 to 3sec, which corresponds to rich chamber characters.
        self.reverb_length = conf.sample_rate
        self.device = device

        # impulse response of reverb.
        self.fir = nn.Parameter(
            torch.rand(1, self.reverb_length, dtype=torch.float32).to(self.device) * 2 - 1,
            requires_grad=True,
        )

        # Initialized drywet to around 26%.
        # but equal-loudness crossfade between identity impulse and fir reverb impulse is not implemented yet.
        self.drywet = nn.Parameter(
            torch.tensor([-1.0], dtype=torch.float32).to(self.device), requires_grad=True
        )

        # Initialized decay to 5, to make t60 = 1sec.
        self.decay = nn.Parameter(
            torch.tensor([3.0], dtype=torch.float32).to(self.device), requires_grad=True
        )

    def forward(self, input_signal):
        """
        Compute FIR Reverb
        Input:
            z['audio_synth'] : batch of time-domain signals
        Output:
            output_signal : batch of reverberated signals
        """

        # Send batch of input signals in time domain to frequency domain.
        # Appropriate zero padding is required for linear convolution.
        zero_pad_input_signal = nn.functional.pad(input_signal, (0, self.fir.shape[-1] - 1))
        INPUT_SIGNAL = torch.view_as_real(fft.rfft(zero_pad_input_signal, dim=1))

        # Build decaying impulse response and send it to frequency domain.
        # Appropriate zero padding is required for linear convolution.
        # Dry-wet mixing is done by mixing impulse response, rather than mixing at the final stage.

        """ TODO 
        Not numerically stable decay method?
        """
        decay_envelope = torch.exp(
            -(torch.exp(self.decay) + 2)
            * torch.linspace(0, 1, self.reverb_length, dtype=torch.float32).to(self.device)
        )
        decay_fir = self.fir * decay_envelope

        ir_identity = torch.zeros(1, decay_fir.shape[-1]).to(self.device)
        ir_identity[:, 0] = 1

        """ TODO
        Equal-loudness(intensity) crossfade between to ir.
        """
        final_fir = (
                torch.sigmoid(self.drywet) * decay_fir + (1 - torch.sigmoid(self.drywet)) * ir_identity
        )
        zero_pad_final_fir = nn.functional.pad(final_fir, (0, input_signal.shape[-1] - 1))

        FIR = torch.view_as_real(fft.rfft(zero_pad_final_fir, dim=1))

        # Convolve and inverse FFT to get original signal.
        OUTPUT_SIGNAL = torch.zeros_like(INPUT_SIGNAL).to(self.device)
        OUTPUT_SIGNAL[:, :, 0] = (
                INPUT_SIGNAL[:, :, 0] * FIR[:, :, 0] - INPUT_SIGNAL[:, :, 1] * FIR[:, :, 1]
        )
        OUTPUT_SIGNAL[:, :, 1] = (
                INPUT_SIGNAL[:, :, 0] * FIR[:, :, 1] + INPUT_SIGNAL[:, :, 1] * FIR[:, :, 0]
        )

        output_signal = fft.irfft(OUTPUT_SIGNAL, dim=1)
        diff = output_signal.shape[1] - input_signal.shape[1]
        output_signal = output_signal[:, diff // 2:-(diff - diff // 2), 0]

        return output_signal


class Reverb(nn.Module):
    def __init__(self, conf=default, initial_wet=0, initial_decay=5):
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
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, lenx - self.length))

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
