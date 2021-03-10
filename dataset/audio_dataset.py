import glob
import os

import torch
import torch.nn.functional as F  # noqa
import torchaudio
from torch.utils.data import Dataset  # noqa
from tqdm import tqdm

from config.default import Config

default = Config()


# TODO: save pitches, loudnesses, and harmonicities to accelerate training.
#       We can drop the encoder part for training, unless we use z-encoder
class AudioData(Dataset):
    def __init__(self, conf=default, clear=False):
        dataset_path = conf.data_dir + '/audio_dataset.pth'
        if os.path.exists(dataset_path) and not clear:
            print('Loading presaved dataset...')
            self.audios = torch.load(dataset_path)
            return

        files = glob.glob(conf.data_dir + '/**/*.wav')
        files += glob.glob(conf.data_dir + '/**/*.mp3')
        if len(files) == 0:
            raise ValueError('No valid audio files found!')

        audios = []
        for f in tqdm(files, desc='Processing audio files...'):
            y, sr = torchaudio.load(f)  # noqa

            # make mono if necessary
            if len(y.shape) == 2:
                if y.shape[0] == 1:
                    y = y[0]
                else:
                    y = y.mean(dim=0)

            # Resample if necessary
            rs = torchaudio.transforms.Resample(sr, conf.sample_rate)
            rs.eval()
            with torch.no_grad():
                y = rs(y)

            # pad to a multiple of hop_size
            pad = len(y) % conf.hop_length
            y = F.pad(y, (pad // 2, pad - pad // 2))

            # modify example duration so that it is a multiple of hop_size
            duration = int(conf.example_duration * conf.sample_rate)
            diff = duration % conf.hop_length
            duration -= diff
            overlap = int(conf.example_overlap * conf.sample_rate)
            diff = duration % conf.hop_length
            overlap -= diff
            # TODO: make the above into a rounding function

            # make overlapping examples
            y = y.unfold(0, duration, overlap)
            audios.append(y)

        self.audios = torch.cat(audios)

        # save dataset
        torch.save(self.audios, dataset_path)

    def __getitem__(self, index):
        return self.audios[index]

    def __len__(self):
        return self.audios.shape[0]
