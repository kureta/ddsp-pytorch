from collections import OrderedDict
from pathlib import Path

import torch


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
