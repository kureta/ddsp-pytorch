import threading

import jack
import numpy as np


class JackIO(threading.Thread):
    def __init__(self, buffer, client_name='zak-rt'):
        super().__init__()
        self.client = jack.Client(client_name)
        self.buffer = buffer

        if self.client.status.server_started:
            print('JACK server started')
        if self.client.status.name_not_unique:
            print('unique name {0!r} assigned'.format(self.client.name))

        self.event = threading.Event()
        self.client.set_process_callback(self.process)
        self.client.set_shutdown_callback(self.process)

        # create one port
        self.client.inports.register('input_1')

        # prepare buffer for pitch tracking
        self.buffer = np.frombuffer(buffer, dtype='float32')

    def process(self, _frames):
        for i in self.client.inports:
            self.buffer[:] = i.get_array()

    def join(self, **kwargs):
        print('JACK shutdown!')
        self.event.set()
        super(JackIO, self).join(**kwargs)

    def setup(self):
        # When entering this with-statement, client.activate() is called.
        # This tells the JACK server that we are ready to roll.
        # Our process() callback will start running now.

        # Connect the ports.  You can't do this before the client is activated,
        # because we can't make connections to clients that aren't running.
        # Note the confusing (but necessary) orientation of the driver backend
        # ports: playback ports are "input" to the backend, and capture ports
        # are "output" from it.

        capture = self.client.get_ports(is_audio=True, is_output=True, is_physical=True)
        if not capture:
            raise RuntimeError('No physical capture ports')

        for src, dest in zip(capture, self.client.inports):
            self.client.connect(src, dest)

        playback = self.client.get_ports(is_physical=True, is_input=True)
        if not playback:
            raise RuntimeError('No physical playback ports')

    def task(self):
        print('Press Ctrl+C to stop')
        try:
            self.event.wait()
        except KeyboardInterrupt:
            print('\nInterrupted by user')

    def run(self):
        with self.client:
            self.setup()
            self.task()
        # When the above with-statement is left (either because the end of the
        # code block is reached, or because an exception was raised inside),
        # client.deactivate() and client.close() are called automatically.
