import torch
import torch.nn as nn

from config.default import Config
from model.ddsp.filtered_noise import FilteredNoise
from model.ddsp.harmonic_oscillator import OscillatorBank
from model.ddsp.reverb import Reverb

default = Config()


class MLP(nn.Module):
    def __init__(self, n_input, n_units, n_layer, relu=nn.LeakyReLU):
        super().__init__()
        self.n_layer = n_layer
        self.n_input = n_input
        self.n_units = n_units

        self.add_module(
            "mlp_layer1",
            nn.Sequential(
                nn.Linear(n_input, n_units),
                nn.LayerNorm(normalized_shape=n_units),
                relu(),
            ),
        )

        for i in range(2, n_layer + 1):
            self.add_module(
                f"mlp_layer{i}",
                nn.Sequential(
                    nn.Linear(n_units, n_units),
                    nn.LayerNorm(normalized_shape=n_units),
                    relu(),
                ),
            )

    def forward(self, x):
        for i in range(1, self.n_layer + 1):
            x = self.__getattr__(f"mlp_layer{i}")(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config=default):
        super().__init__()

        self.config = config

        self.mlp_f0 = MLP(
            n_input=1,
            n_units=config.decoder_mlp_units,
            n_layer=config.decoder_mlp_layers
        )
        self.mlp_loudness = MLP(
            n_input=1,
            n_units=config.decoder_mlp_units,
            n_layer=config.decoder_mlp_layers
        )
        self.mlp_harmonicity = MLP(
            n_input=1,
            n_units=config.decoder_mlp_units,
            n_layer=config.decoder_mlp_layers
        )

        if config.use_z:
            self.mlp_z = MLP(
                n_input=config.z_units,
                n_units=config.decoder_mlp_units,
                n_layer=config.decoder_mlp_layers
            )
            self.num_mlp = 4
        else:
            self.num_mlp = 3

        self.gru = nn.GRU(
            input_size=self.num_mlp * config.decoder_mlp_units,
            hidden_size=config.decoder_gru_units,
            num_layers=config.decoder_gru_layers,
            batch_first=True,
        )

        self.mlp_gru = MLP(
            n_input=config.decoder_gru_units + self.num_mlp,
            n_units=config.decoder_mlp_units,
            n_layer=config.decoder_mlp_layers,
        )

        # one element for overall loudness
        self.dense_harmonic = nn.Linear(config.decoder_mlp_units, config.n_harmonics + 1)
        self.dense_filter = nn.Linear(config.decoder_mlp_units, config.n_noise_filters)

    def forward(self, batch, hidden=None):
        f0 = batch['normalized_cents']
        loudness = batch['loudness']
        harmonicity = batch['harmonicity']

        if self.config.use_z:
            z = batch['z']
            latent_z = self.mlp_z(z)

        latent_f0 = self.mlp_f0(f0)
        latent_loudness = self.mlp_loudness(loudness)
        latent_harmonicity = self.mlp_harmonicity(harmonicity)

        if self.config.use_z:
            latent = torch.cat((latent_f0, latent_z, latent_loudness, latent_harmonicity), dim=-1)
        else:
            latent = torch.cat((latent_f0, latent_loudness, latent_harmonicity), dim=-1)

        if hidden is not None:
            latent, h = self.gru(latent, hidden)
        else:
            latent, h = self.gru(latent)

        latent = torch.cat((latent, f0, loudness, harmonicity), dim=-1)
        latent = self.mlp_gru(latent)

        harm = self.modified_sigmoid(self.dense_harmonic(latent))
        harm_amps = harm[..., 1:]
        total_harm_amp = harm[..., :1]

        noise_distribution = self.dense_filter(latent)
        noise_distribution = self.modified_sigmoid(noise_distribution - 5)

        if hidden is not None:
            return dict(f0=batch["f0"], c=harm_amps, hidden=h, H=noise_distribution, a=total_harm_amp), hidden
        return dict(f0=batch["f0"], c=harm_amps, hidden=h, H=noise_distribution, a=total_harm_amp)

    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a


class DDSPDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.ddsp = OscillatorBank()
        self.noise = FilteredNoise()
        self.reverb = Reverb()

    def forward(self, z):
        ctrl = self.decoder(z)
        harmonics = self.ddsp(ctrl)
        noise = self.noise(ctrl)

        signal = harmonics + noise
        signal = self.reverb(signal)

        return signal

    # TODO: Every module should be responsible for keeping track of their own
    #       internal state in a `forward_live` method
    def forward_live(self, z, hidden):
        ctrl, hidden = self.decoder(z, hidden)
        harmonics = self.ddsp.live(ctrl)
        noise = self.noise(ctrl)

        audio_hat = harmonics + noise
        audio_hat = self.reverb.live_forward(audio_hat)

        return audio_hat.cpu().squeeze(0).numpy(), hidden
