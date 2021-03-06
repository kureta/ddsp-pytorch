import threading
from time import time

import jack
import numpy as np
import torch

from config.default import Config
from model.autoencoder.autoencoder import AutoEncoder
from rt.utils import load_checkpoint

default = Config()

# TODO: this can easily be expanded to stereo by processing in batches of 2
# Prepare zak
zak = AutoEncoder(default)
zak.decoder.load_state_dict(load_checkpoint(7))
zak.eval()
zak = zak.cuda()

# Prepare network inputs
hidden = torch.randn(zak.decoder.controller.gru.num_layers, 1, 512).cuda()
input_buffer = np.zeros(4096, dtype='float32')

# Run once to buld the compuation graph
with torch.no_grad():
    _ = zak.forward_live(input_buffer, hidden)

# Prepare jack client
client = jack.Client('zak-rt')

if client.status.server_started:
    print('JACK server started')
if client.status.name_not_unique:
    print('unique name {0!r} assigned'.format(client.name))

event = threading.Event()


@client.set_process_callback
def process(frames):
    assert len(client.inports) == len(client.outports)
    assert frames == client.blocksize
    input_buffer[:-frames] = input_buffer[frames:]
    for i in client.inports:
        current_buffer = np.frombuffer(i.get_buffer(), dtype='float32')
        input_buffer[-frames:] = current_buffer
    for o in client.outports:
        now = time()
        with torch.no_grad():
            out_signal, hidden[...] = zak.forward_live(input_buffer, hidden)
            o.get_buffer()[:] = out_signal[-frames:]
        dur = time() - now
        if dur >= frames / 44100:
            print('missed a frame')


@client.set_shutdown_callback
def shutdown(status, reason):
    print('JACK shutdown!')
    print('status:', status)
    print('reason:', reason)
    event.set()


# create two ports
client.inports.register('input_1')
client.outports.register('output_1')

with client:
    capture = client.get_ports(is_physical=True, is_output=True)
    if not capture:
        raise RuntimeError('No physical capture ports')

    for src, dest in zip(capture, client.inports):
        client.connect(src, dest)

    playback = client.get_ports(is_physical=True, is_input=True)
    if not playback:
        raise RuntimeError('No physical playback ports')

    for src, dest in zip(client.outports, playback):
        client.connect(src, dest)

    print('Press Ctrl+C to stop')
    try:
        event.wait()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
