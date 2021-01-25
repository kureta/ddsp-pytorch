import numpy as np
import torch

from components.loudness_extractor import LoudnessExtractor
from components.ptcrepe.ptcrepe.crepe import CREPE
from rt.nodes.base_nodes import BaseNode


class PitchTracker(BaseNode):
    def __init__(self, buffer, freq, confidence, amp):
        super().__init__()

        self.cr = self.le = None
        self.buffer = buffer
        self.freq = freq
        self.confidence = confidence
        self.amp = amp

    def setup(self):
        self.cr = CREPE('medium').cuda()
        self.le = LoudnessExtractor().cuda()
        self.buffer = np.frombuffer(self.buffer, dtype='float32')

    def task(self):
        with torch.no_grad():
            buffer = torch.from_numpy(self.buffer[np.newaxis, :]).cuda()
            _, freq, confidence, _ = self.cr.predict(
                buffer,
                48000,
                viterbi=False,
                center=True,
                step_size=10,
                batch_size=128
            )
            amp = self.le(buffer)

        # TODO: alignment or synchronization or something.
        self.freq.value = freq[-1]
        self.confidence.value = confidence[-1]
        self.amp.value = amp[0, -1]
