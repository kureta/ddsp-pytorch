import numpy as np
import torch

from model.ddsp.loudness_extractor import LoudnessExtractor
from model.ptcrepe.crepe import CREPE
from rt.nodes.base_nodes import BaseNode


class PitchTracker(BaseNode):
    def __init__(self, buffer, freq, confidence, loudness):
        super().__init__()

        self.cr = self.le = None
        self.buffer = buffer
        self.freq = freq
        self.confidence = confidence
        self.loudness = loudness

    def setup(self):
        self.cr = CREPE('medium').cuda()
        self.le = LoudnessExtractor().cuda()
        self.cr.eval()
        self.le.eval()
        self.buffer = np.frombuffer(self.buffer, dtype='float32')

    def task(self):
        with torch.no_grad():
            buffer = torch.from_numpy(self.buffer[np.newaxis, :]).cuda()
            freq, confidence, _ = self.cr.predict(
                buffer,
                48000,
                center=True,
                step_size=10,
                batch_size=128
            )
            loudness = self.le({'audio': buffer})

        # TODO: alignment or synchronization or something.
        self.freq.value = freq[-1]
        self.confidence.value = confidence[-1]
        self.loudness.value = loudness[0, -1]
