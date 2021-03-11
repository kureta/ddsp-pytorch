from dataclasses import dataclass


@dataclass
class Config:
    data_dir = '/home/kureta/Music/violin'
    example_duration = 1  # in seconds
    example_overlap = 0.5
    sample_rate = 44100
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    n_mfcc = 42
    crepe_capacity = 'full'
    f_min = 20.
    f_max = 8000.
    n_harmonics = 100
    n_noise_filters = 65
    use_z = False
    encoder_gru_units = 512
    z_units = 16
    decoder_mlp_units = 512
    decoder_mlp_layers = 3
    decoder_gru_units = 512
    decoder_gru_layers = 1
    batch_size = 16
