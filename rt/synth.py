import ctypes
import signal
import threading
from multiprocessing import Event, set_start_method
from multiprocessing.sharedctypes import RawArray, RawValue  # noqa

import jack
import numpy as np
import torch
from pythonosc import dispatcher, osc_server

from model.ddsp.harmonic_oscillator import OscillatorBank
from rt.nodes.base_nodes import BaseNode

SAMPLE_RATE = 48000
HOP_SIZE = 512
NUM_HARMONICS = 100


class JackOut(threading.Thread):
    def __init__(self, output_buffer: RawArray, flag: RawValue, client_name='zak-rt'):
        super().__init__()
        self.client = jack.Client(client_name)

        if self.client.status.server_started:
            print('JACK server started')
        if self.client.status.name_not_unique:
            print('unique name {0!r} assigned'.format(self.client.name))

        self.event = threading.Event()
        self.client.set_process_callback(self.process)
        self.client.set_shutdown_callback(self.join)

        # create one port
        self.client.outports.register('output_01')

        # prepare buffer for pitch tracking
        self.output_buffer = np.frombuffer(output_buffer, dtype='float32')
        self.flag = flag

    def process(self, _frames):
        self.flag.value = True
        for o in self.client.outports:
            o.get_buffer()[:] = self.output_buffer

    def join(self, **kwargs):
        print('JACK shutdown!')
        self.event.set()
        super().join(**kwargs)

    def setup(self):
        # When entering this with-statement, client.activate() is called.
        # This tells the JACK server that we are ready to roll.
        # Our process() callback will start running now.

        # Connect the ports.  You can't do this before the client is activated,
        # because we can't make connections to clients that aren't running.
        # Note the confusing (but necessary) orientation of the driver backend
        # ports: playback ports are "input" to the backend, and capture ports
        # are "output" from it.

        playback = self.client.get_ports(is_audio=True, is_input=True, is_physical=True)
        if not playback:
            raise RuntimeError('No physical playback ports')

        for src, dest in zip(self.client.outports, playback):
            self.client.connect(src, dest)

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
        # When the above with-statement is left (either because the end of the
        # code block is reached, or because an exception was raised inside),
        # client.deactivate() and client.close() are called automatically.


class Synth(BaseNode):
    def __init__(self, freq: RawValue, amp: RawValue, stretch: RawArray,
                 harmonics: RawArray, output_buffer: RawArray, flag: RawValue):
        super().__init__()
        self.output_buffer = output_buffer
        self.harmonics = np.frombuffer(harmonics, dtype='float32')
        for idx, _ in enumerate(self.harmonics):
            self.harmonics[idx] = 1 / (1 + idx)
        # self.harmonics[0] = 1.
        self.amp = amp
        self.freq = freq
        self.stretch = stretch
        self.flag = flag

        self.t_freq = self.t_amp = self.t_harmonics = self.t_harm_stretch = self.additive = None

    def setup(self):
        self.t_freq = torch.zeros(1, device='cuda')
        self.t_amp = torch.zeros(1, device='cuda')
        self.t_harm_stretch = torch.zeros(1, device='cuda')
        self.t_harmonics = torch.zeros(NUM_HARMONICS, device='cuda')
        self.additive = OscillatorBank(n_harmonics=NUM_HARMONICS, sample_rate=SAMPLE_RATE, hop_size=HOP_SIZE).to('cuda')

    def task(self):
        if not self.flag.value:
            return
        self.flag.value = False
        self.t_freq[0] = self.freq.value
        self.t_amp[0] = self.amp.value
        self.t_harm_stretch[0] = self.stretch.value
        self.t_harmonics = torch.from_numpy(self.harmonics).cuda()

        with torch.no_grad():
            audio = self.additive(self.t_freq, self.t_amp, self.t_harmonics, self.t_harm_stretch)
        self.output_buffer[:] = audio.cpu().numpy()


class OSCServer(threading.Thread):
    def __init__(self, freq: RawValue, amp: RawValue, stretch: RawValue):
        print('server start')
        super().__init__()
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', 8000), self.dispatcher)

        self.freq = freq
        self.amp = amp
        self.stretch = stretch

        self.dispatcher.map('/controls/freq', self.on_freq)  # noqa
        self.dispatcher.map('/controls/amp', self.on_amp)  # noqa
        self.dispatcher.map('/controls/stretch', self.on_stretch)  # noqa
        self.dispatcher.set_default_handler(self.on_unknown_message)

    @staticmethod
    def on_unknown_message(addr, *values):
        print(f'addr: {addr}', f'values: {values}')

    def on_freq(self, _, value):
        self.freq.value = value

    def on_amp(self, _, value):
        self.amp.value = value

    def on_stretch(self, _, value):
        self.stretch.value = value

    def run(self):
        self.server.serve_forever()

    def join(self, **kwargs):
        self.server.shutdown()
        super(OSCServer, self).join(**kwargs)


class App:
    def __init__(self):
        set_start_method('spawn', force=True)
        output_buffer = RawArray(ctypes.c_float, HOP_SIZE * [0.])
        freq = RawValue(ctypes.c_float)
        amp = RawValue(ctypes.c_float)
        stretch = RawValue(ctypes.c_float)
        flag = RawValue(ctypes.c_bool)
        harmonics = RawArray(ctypes.c_float, NUM_HARMONICS * [0.])
        freq.value = 220.
        amp.value = .9
        stretch.value = 0.
        flag.value = False

        self.jack = JackOut(output_buffer, flag)
        self.synth = Synth(freq, amp, stretch, harmonics, output_buffer, flag)
        self.control = OSCServer(freq, amp, stretch)

        self.exit = Event()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.jack.start()
        self.synth.start()
        self.control.start()

        signal.signal(signal.SIGINT, self.on_keyboard_interrupt)

        self.exit.wait()

    def on_keyboard_interrupt(self, sig, _):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.exit_handler()

    def exit_handler(self):
        self.jack.join()
        self.synth.join()
        self.control.join()

        exit(0)


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
