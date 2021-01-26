import glob
import os

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

# TODO: step_size, frame_length, frame_resolution are all based on one variable.
#       they are all hop sizes.
#       frame_reslution = hop_size in seconds
#       step_size = hop_size in milliseconds
#       frame_length = hop_size in samples
#       just rename everything to hop_size ffs.
from tqdm import tqdm

from model.ddsp.loudness_extractor import LoudnessExtractor
from model.ptcrepe.crepe import CREPE


class AudioData(Dataset):
    def __init__(self, folder_path, sample_rate=48000):
        dataset_path = folder_path + '/audio_dataset.pth'
        if os.path.exists(dataset_path):
            print('Loading presaved dataset...')
            self.audios = torch.load(dataset_path)
            return

        files = glob.glob(folder_path + '/**/*.wav')
        files += glob.glob(folder_path + '/**/*.mp3')
        if len(files) == 0:
            raise ValueError('No valid audio files found!')

        audios = []
        print('Processing audio files...')
        for f in tqdm(files):
            y, sr = torchaudio.load(f, )
            if len(y.shape) == 2:
                if y.shape[0] == 1:
                    y = y[0]
                else:
                    y = y.mean(dim=0)
            # normalize
            y = y / (torch.abs(y).max())
            rs = torchaudio.transforms.Resample(sr, sample_rate)
            rs.eval()
            y = rs(y)
            y = F.pad(y, (1024, 1024))
            # make non-overlapping windows
            y = y.unfold(0, 2048, 2048)
            audios.append(y)
        # concatenate all tracks
        self.audios = torch.cat(audios)
        torch.save(self.audios, dataset_path)

    def __getitem__(self, index):
        return self.audios[index]

    def __len__(self):
        return self.audios.shape[0]


class LabeledAudioData(Dataset):
    def __init__(self, folder_path, sample_rate=48000, duration=24):
        self.duration = duration
        dataset_path = folder_path + '/labeled_dataset.pth'
        if os.path.exists(dataset_path):
            print('Loading presaved dataset...')
            self.freq, self.confidence, self.loudness = torch.load(dataset_path)
            self.audio = AudioData(folder_path, sample_rate).audios
            self.create_examples()
            return

        crepe = CREPE(model_capacity='full').eval().cuda()
        lex = LoudnessExtractor(sample_rate).eval().cuda()
        audio = AudioData(folder_path, sample_rate)
        data_loader = DataLoader(audio, batch_size=128, shuffle=False)
        freqs = []
        confidences = []
        loudnesses = []
        current = 0
        for batch in tqdm(data_loader):
            current += 128
            batch = batch.cuda()
            with torch.no_grad():
                freq, confidence, _ = crepe.predict(batch, sample_rate)
                loudness = lex(batch)
            freqs.append(freq.cpu())
            confidences.append(confidence.cpu())
            loudnesses.append(loudness.cpu())

        del crepe
        del lex

        self.audio = audio.audios
        self.freq = torch.cat(freqs)
        self.confidence = torch.cat(confidences)
        self.loudness = torch.cat(loudnesses)
        torch.save((self.freq, self.confidence, self.loudness), dataset_path)

        self.create_examples()

    def create_examples(self):
        # Create examples
        self.audio = self.audio.unfold(0, self.duration, self.duration // 2).permute(0, 2, 1)
        self.freq = self.freq.unfold(0, self.duration, self.duration // 2)
        self.confidence = self.confidence.unfold(0, self.duration, self.duration // 2)
        self.loudness = self.loudness.unfold(0, self.duration, self.duration // 2)

    def __getitem__(self, index):
        return {
            'audio': self.audio[index],
            'f0': self.freq[index],
            'confidence': self.confidence[index],
            'loudness': self.loudness[index],
        }

    def __len__(self):
        return self.audio.shape[0]
