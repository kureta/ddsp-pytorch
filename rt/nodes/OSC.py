from time import sleep

import numpy as np
from pythonosc import udp_client

from rt.nodes.base_nodes import BaseNode


class OSC(BaseNode):
    def __init__(self, freq, confidence, amp, harmonics):
        print('server start')
        super().__init__()

        self.client = None
        self.freq = freq
        self.confidence = confidence
        self.amp = amp
        self.harmonics = harmonics

    def setup(self):
        ip = '127.0.0.1'
        port = 57120
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.harmonics = np.frombuffer(self.harmonics, dtype='float32')

    def task(self):
        if self.confidence.value < 0.5:
            self.freq.value = 0.0
        self.client.send_message('/crepe/freq', self.freq.value)
        self.client.send_message('/crepe/amp', self.amp.value)
        self.client.send_message('/crepe/harmonics', self.harmonics.tolist())
        sleep(1/128)
