from omegaconf import OmegaConf
import torch
import numpy as np

from rt.nodes.base_nodes import BaseNode
from train.network.autoencoder.autoencoder import AutoEncoder

CONF_PATH = '/home/kureta/Documents/ddsp-pytorch-pretrained/weight/200220.pth.yaml'
WEIGHTS_PATH = '/home/kureta/Documents/ddsp-pytorch-pretrained/weight/200220.pth'


class Decoder(BaseNode):
    def __init__(self, freq, loudness, amp, harmonics):
        super().__init__()
        config = OmegaConf.load(CONF_PATH)
        net = AutoEncoder(config)
        net.eval()
        self.decoder = net.decoder
        self.freq = freq
        self.loudness = loudness
        self.hidden = np.random.randn(1, 1, 512).astype('float32')
        self.loudness_t = np.zeros((1, 1), dtype='float32')
        self.freq_t = np.zeros((1, 1), dtype='float32')
        self.amp = amp
        self.harmonics = harmonics

    def setup(self):
        self.decoder = self.decoder.cuda()
        self.loudness_t = torch.from_numpy(self.loudness_t).cuda()
        self.freq_t = torch.from_numpy(self.freq_t).cuda()
        self.hidden = torch.from_numpy(self.hidden).cuda()
        self.harmonics = np.frombuffer(self.harmonics, dtype='float32')

    def task(self):
        with torch.no_grad():
            self.loudness_t[...] = self.loudness.value
            self.freq_t[...] = self.freq.value

            z = {
                'f0': self.freq_t,
                'loudness': self.loudness_t,
                'hidden': self.hidden,
            }

            result = self.decoder(z)
            self.hidden[...] = result['hidden']

            self.harmonics[:] = result['c'][0, 0].cpu().numpy()
            self.amp.value = result['a'][0, 0].cpu().numpy()
