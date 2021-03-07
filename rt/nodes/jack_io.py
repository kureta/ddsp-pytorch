import threading

import jack
import numpy as np


class JackIO(threading.Thread):
    def __init__(self, input_buffer, output_buffer, flag):
        super().__init__()
        self.client = jack.Client('zak_rt')

        if self.client.status.server_started:
            print('JACK server started')
        if self.client.status.name_not_unique:
            print('unique name {0!r} assigned'.format(self.client.name))

        self.event = threading.Event()
        self.client.set_process_callback(self.process)
        self.client.set_shutdown_callback(self.join)

        # create one port pair
        self.client.outports.register('output_01')
        self.client.inports.register('input_01')

        # prepare buffer for pitch tracking
        self.output_buffer = np.frombuffer(output_buffer, dtype='float32')
        self.input_buffer = np.frombuffer(input_buffer, dtype='float32')
        self.flag = flag

    def process(self, _frames):
        self.flag.value = True
        for i in self.client.inports:
            # swap buffers
            self.input_buffer[2048:] = self.input_buffer[-2048:]
            # fill new buffer
            buff = np.frombuffer(i.get_buffer(), dtype='float32')
            self.input_buffer[-2048:] = buff[-2048:]
        for o in self.client.outports:
            o.get_buffer()[:] = self.output_buffer

    def join(self, **kwargs):
        print('JACK shutdown!')
        self.event.set()
        super().join(**kwargs)

    def setup(self):
        playback = self.client.get_ports(is_audio=True, is_input=True, is_physical=True)
        if not playback:
            raise RuntimeError('No physical playback ports')

        for dest in playback:
            self.client.connect(self.client.outports[0], dest)

        recording = self.client.get_ports(is_audio=True, is_output=True, is_physical=True)
        if not recording:
            raise RuntimeError('No physical input ports')

        for dest in recording:
            self.client.connect(dest, self.client.inports[0])

    def task(self):
        print('JACK is processing...')
        try:
            self.event.wait()
        except KeyboardInterrupt:
            print('\nInterrupted by user')

    def run(self):
        with self.client:
            self.setup()
            self.task()
