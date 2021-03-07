import torch
import numpy as np

from rt.nodes.base_nodes import BaseNode
from model.autoencoder.encoder import Encoder
from model.autoencoder.decoder import Decoder
from model.ddsp.harmonic_oscillator import OscillatorBank


class AutoEncoder(BaseNode):
    def __init__(self, audio_in, audio_out):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.ddsp = OscillatorBank()
        for module in [self.encoder, self.decoder, self.ddsp]:
            module.eval()

        self.audio_in = audio_in
        self.audio_out = audio_out
        # This module should receive the last 2048-256 samples of the previous frame
        # concat with the current frame - last 256 samples
        # Easier way to achive this is to just get last 2 frames and
        # drop first and last 256 samples
        self.audio_in_t = np.zeros(4096, dtype='float32')

    def setup(self):
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()
        self.ddsp = self.ddsp.cuda()

        self.audio_out = np.frombuffer(self.audio_out, dtype='float32')
        self.audio_in = np.frombuffer(self.audio_in, dtype='float32')
        self.audio_in_t = torch.from_numpy(self.audio_in_t).cuda()

    def task(self):
        with torch.no_grad():
            audio_in = torch.from_numpy(self.audio_in).unsqueeze(0).cuda()
            # We are dropping those samples here
            z = self.encoder(audio_in[:, 256:-256])
            ctrl = self.decoder(z)
            audio_hat = self.ddsp.live(ctrl)

            self.audio_out[...] = audio_hat.cpu().squeeze(0).numpy()
