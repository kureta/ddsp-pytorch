from dataclasses import dataclass


@dataclass
class Config:
    batch_size = 64
    bidirectional = False
    ckpt = '/home/kureta/.cache/ddsp-pytorch/checkpoint.pth'
    crepe = 'full'
    experiment_name = 'DDSP_violin'
    f0_threshold = 0.5
    frame_resolution = 0.004
    gpu = 0
    gru_units = 512
    loss = 'mss'
    lr = 0.001
    lr_decay = 0.98
    lr_min: 1.0e-07
    lr_scheduler = 'multi'
    metric = 'mss'
    mlp_layers = 3
    mlp_units = 512
    n_fft = 2048
    n_freq = 65
    n_harmonics = 101
    n_mels = 128
    n_mfcc = 30
    num_step = 100000
    num_workers = 4
    optimizer = 'radam'
    resume = False
    sample_rate = 16000
    seed = 940513
    tensorboard_dir = '/home/kureta/.cache/ddsp-pytorch/log'
    test = '/home/kureta/.cache/ddsp-pytorch/data/test'
    train = '/home/kureta/.cache/ddsp-pytorch/data/train'
    use_reverb = True
    use_z = False
    valid_waveform_sec = 12
    validation_interval = 1000
    waveform_sec = 1
    z_units = 16
