from time import sleep

from pythonosc import udp_client

from rt.nodes.base_nodes import BaseNode


class OSC(BaseNode):
    def __init__(self, freq, confidence, amp):
        print('server start')
        super().__init__()

        self.client = None
        self.freq = freq
        self.confidence = confidence
        self.amp = amp

    def setup(self):
        ip = '127.0.0.1'
        port = 57120
        self.client = udp_client.SimpleUDPClient(ip, port)

    def task(self):
        self.client.send_message('/crepe/freq', self.freq.value)
        self.client.send_message('/crepe/confidence', self.confidence.value)
        self.client.send_message('/crepe/amp', self.amp.value)
        sleep(1/128)
