from pathlib import Path

import numpy as np
import torch

from rt.nodes.base_nodes import BaseNode
from train.train import AutoEncoder


def load_checkpoint(version):
    file = Path(
        Path.cwd(),
        'lightning_logs',
        f'version_{version}',
        'checkpoints',
    ).glob('*.ckpt')
    file = sorted(list(file), key=lambda x: int(x.name.split('-')[0].split('=')[1]))
    file = file[-1]

    state_dict = torch.load(file)['state_dict']
    new_state = {}
    for key in state_dict.keys():
        if key.startswith('model'):
            new_key = key[6:]
            new_state[new_key] = state_dict[key]

    return new_state


class Zak(BaseNode):
    def __init__(self, audio_in, audio_out, flag):
        super().__init__()
        self.flag = flag
        self.autoencoder = AutoEncoder()
        self.autoencoder.load_state_dict(load_checkpoint(9))
        self.autoencoder.eval()

        self.audio_in = audio_in
        self.audio_out = audio_out
        # This module should receive the last 2048-256 samples of the previous frame
        # concat with the current frame - last 256 samples
        # Easier way to achive this is to just get last 2 frames and
        # drop first and last 256 samples
        self.hidden = torch.randn(1, 1, 512)
        self.times = None

    def setup(self):
        self.autoencoder = self.autoencoder.cuda()
        self.hidden = self.hidden.cuda()

        self.audio_out = np.frombuffer(self.audio_out, dtype='float32')
        self.audio_in = np.frombuffer(self.audio_in, dtype='float32')
        with torch.no_grad():
            self.audio_out[...], self.hidden = self.autoencoder.forward_live(self.audio_in, self.hidden)

    def task(self):
        if not self.flag.value:
            print('waiting new frame')
            return
        self.flag.value = True
        with torch.no_grad():
            self.audio_out[...], self.hidden = self.autoencoder.forward_live(self.audio_in, self.hidden)
