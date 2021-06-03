from dataclasses import dataclass


# v11 = cello
# v12 = violin
# v16 = viola
# v17 = flute
@dataclass
class Config:
    data_dir = '/home/kureta/Music/flute'
    example_duration = 2  # in seconds
    example_overlap = 0.5
    sample_rate = 44100
    n_fft = 2048
    hop_length = 512
    crepe_capacity = 'full'

    n_harmonics = 180
    n_noise_filters = 195
    decoder_mlp_units = 512
    decoder_mlp_layers = 3
    decoder_gru_units = 512
    decoder_gru_layers = 1
    batch_size = 16
