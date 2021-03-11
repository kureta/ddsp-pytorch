import torch
import torch.nn as nn

from model.ddsp.filtered_noise import FilteredNoise
from model.ddsp.harmonic_oscillator import OscillatorBank
from model.ddsp.reverb import Reverb


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


class Controller(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.config = conf

        self.mlp_f0 = MLP(
            n_input=1,
            n_units=conf.decoder_mlp_units,
            n_layer=conf.decoder_mlp_layers
        )
        self.mlp_loudness = MLP(
            n_input=1,
            n_units=conf.decoder_mlp_units,
            n_layer=conf.decoder_mlp_layers
        )
        self.mlp_harmonicity = MLP(
            n_input=1,
            n_units=conf.decoder_mlp_units,
            n_layer=conf.decoder_mlp_layers
        )

        if conf.use_z:
            self.mlp_z = MLP(
                n_input=conf.z_units,
                n_units=conf.decoder_mlp_units,
                n_layer=conf.decoder_mlp_layers
            )
            self.num_mlp = 4
        else:
            self.num_mlp = 3

        self.gru = nn.GRU(
            input_size=self.num_mlp * conf.decoder_mlp_units,
            hidden_size=conf.decoder_gru_units,
            num_layers=conf.decoder_gru_layers,
            batch_first=True,
        )

        self.mlp_gru = MLP(
            n_input=conf.decoder_gru_units + self.num_mlp,
            n_units=conf.decoder_mlp_units,
            n_layer=conf.decoder_mlp_layers,
        )

        # one element for overall loudness
        self.dense_harmonic = nn.Linear(conf.decoder_mlp_units, conf.n_harmonics)
        self.dense_loudness = nn.Linear(conf.decoder_mlp_units, 1)
        self.dense_filter = nn.Linear(conf.decoder_mlp_units, conf.n_noise_filters)

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

        harm_amps = self.modified_sigmoid(self.dense_harmonic(latent))
        total_harm_amp = self.modified_sigmoid(self.dense_loudness(latent))

        noise_distribution = self.dense_filter(latent)
        noise_distribution = self.modified_sigmoid(noise_distribution - 5)

        controls = dict(f0=batch["f0"], c=harm_amps, hidden=h, H=noise_distribution, a=total_harm_amp)
        if hidden is not None:
            return controls, hidden
        return controls

    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a


class Decoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.controller = Controller(conf)
        self.harmonics = OscillatorBank(conf)
        self.noise = FilteredNoise(conf)
        self.reverb = Reverb(conf)

    def forward(self, z):
        ctrl = self.controller(z)
        harmonics = self.harmonics(ctrl)
        noise = self.noise(ctrl)

        signal = harmonics + noise
        signal = self.reverb(signal)

        return signal

    # TODO: Every module should be responsible for keeping track of their own
    #       internal state in a `forward_live` method
    def forward_live(self, z, hidden):
        ctrl, hidden = self.controller(z, hidden)
        harmonics = self.harmonics.live(ctrl)
        noise = self.noise(ctrl)

        audio_hat = harmonics + noise
        audio_hat = self.reverb.live_forward(audio_hat)

        return audio_hat.cpu().squeeze(0).numpy(), hidden
