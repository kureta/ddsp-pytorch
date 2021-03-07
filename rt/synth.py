import threading
from collections import OrderedDict
from pathlib import Path

import jack
import numpy as np
import torch

from model.autoencoder.autoencoder import AutoEncoder


def load_checkpoint(version):
    file = Path(
        Path.cwd(),
        'lightning_logs',
        f'version_{version}',
        'checkpoints',
    ).glob('*.ckpt')
    file = sorted(list(file), key=lambda x: int(x.name.split('-')[0].split('=')[1]))
    file = file[-1]

    state_dict = torch.load(file)['state_dict']
    new_state = OrderedDict()
    for key in state_dict.keys():
        if key.startswith('model'):
            new_key = key[6:]
            new_state[new_key] = state_dict[key]

    return new_state


zak = AutoEncoder()
zak.load_state_dict(load_checkpoint(9))
zak.eval()
zak = zak.cuda()

hidden = torch.randn(1, 1, 512).cuda()
input_buffer = np.zeros(4096, dtype='float32')

with torch.no_grad():
    _ = zak.forward_live(input_buffer, hidden)

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
    input_buffer[:2048] = input_buffer[-2048:]
    for i in client.inports:
        shit = np.frombuffer(i.get_buffer(), dtype='float32')
        input_buffer[-2048:] = shit
    for o in client.outports:
        with torch.no_grad():
            o.get_buffer()[:], hidden[...] = zak.forward_live(input_buffer, hidden)


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
    # When entering this with-statement, client.activate() is called.
    # This tells the JACK server that we are ready to roll.
    # Our process() callback will start running now.

    # Connect the ports.  You can't do this before the client is activated,
    # because we can't make connections to clients that aren't running.
    # Note the confusing (but necessary) orientation of the driver backend
    # ports: playback ports are "input" to the backend, and capture ports
    # are "output" from it.

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
