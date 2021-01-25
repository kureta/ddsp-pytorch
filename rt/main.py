import ctypes
import signal
from multiprocessing import Event
from multiprocessing.sharedctypes import RawArray, RawValue  # noqa

from rt.nodes.OSC import OSC
from rt.nodes.pass_through import JackIO
from rt.nodes.pitch_tracker import PitchTracker


class App:
    def __init__(self):
        buffer = RawArray(ctypes.c_float, 2048 * [0.])
        freq = RawValue(ctypes.c_float)
        confidence = RawValue(ctypes.c_float)
        amp = RawValue(ctypes.c_float)

        self.jack = JackIO(buffer)
        self.pt = PitchTracker(buffer, freq, confidence, amp)
        self.osc = OSC(freq, confidence, amp)

        self.exit = Event()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.jack.start()
        self.pt.start()
        self.osc.start()

        signal.signal(signal.SIGINT, self.on_keyboard_interrupt)

        self.exit.wait()

    def on_keyboard_interrupt(self, sig, _):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.exit_handler()

    def exit_handler(self):
        self.jack.join()
        self.pt.join()
        self.osc.join()

        exit(0)


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
