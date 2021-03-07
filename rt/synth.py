import ctypes
import signal
from multiprocessing import Event, set_start_method
from multiprocessing.sharedctypes import RawArray, RawValue  # noqa

from rt.nodes.autoencoder import Zak
from rt.nodes.jack_io import JackIO


class App:
    def __init__(self):
        set_start_method('spawn', force=True)
        output_buffer = RawArray(ctypes.c_float, 2048 * [0.])
        input_buffer = RawArray(ctypes.c_float, 4096 * [0.])

        flag = RawValue(ctypes.c_bool)
        flag.value = False

        self.jack = JackIO(input_buffer, output_buffer, flag)
        self.zak = Zak(input_buffer, output_buffer)

        self.exit = Event()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.jack.start()
        self.zak.start()

        signal.signal(signal.SIGINT, self.on_keyboard_interrupt)

        self.exit.wait()

    def on_keyboard_interrupt(self, sig, _):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.exit_handler()

    def exit_handler(self):
        self.jack.join()
        self.zak.join()

        exit(0)


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
