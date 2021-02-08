import glob
import os

import torch
import torch.fft
import torch.nn.functional as F  # noqa
import torchaudio
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from loss.mss_loss import MSSLoss
from model.autoencoder.decoder import MLP


class AudioData(Dataset):
    def __init__(self, folder_path, sample_rate=44100):
        audio_dataset_path = folder_path + '/audio_dataset.pth'
        spectrum_dataset_path = folder_path + '/spectrum_dataset.pth'
        if os.path.exists(audio_dataset_path) and os.path.exists(spectrum_dataset_path):
            print('Loading presaved dataset...')
            self.audios = torch.load(audio_dataset_path)
            self.magnitude_spectrums = torch.load(spectrum_dataset_path)
            return

        files = glob.glob(folder_path + '/**/*.wav')
        files += glob.glob(folder_path + '/**/*.mp3')
        if len(files) == 0:
            raise ValueError('No valid audio files found!')

        audios = []
        magnitude_spectrums = []
        print('Processing audio files...')
        for f in tqdm(files):
            y, sr = torchaudio.load(f, )
            # convert to mono
            if len(y.shape) == 2:
                if y.shape[0] == 1:
                    y = y[0]
                else:
                    y = y.mean(dim=0)
            # normalize
            y = y / (torch.abs(y).max())
            if sr != sample_rate:
                rs = torchaudio.transforms.Resample(sr, sample_rate)
                rs.eval()
                y = rs(y)
            padding = y.shape[0] % 1024
            y = F.pad(y, (padding // 2, padding - (padding // 2)))
            # make non-overlapping windows
            y = y.unfold(0, sr * 2, sr)
            audios.append(y)
            spec = torch.stft(y, n_fft=1024, hop_length=256, window=torch.hann_window(1024))
            spec = spec.pow(2.).sum(-1).pow(0.5)
            magnitude_spectrums.append(spec)
        # concatenate all tracks
        self.audios = torch.cat(audios)
        self.magnitude_spectrums = torch.cat(magnitude_spectrums)
        torch.save(self.audios, audio_dataset_path)
        torch.save(self.magnitude_spectrums, spectrum_dataset_path)

    def __getitem__(self, index):
        return self.magnitude_spectrums[index].T, self.audios[index]

    def __len__(self):
        return self.audios.shape[0]


class Recon(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_mpl = MLP(513, 256, 3)
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.post_mpl = MLP(256, 513, 3)
        self.output = nn.Linear(513, 513)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        latent = self.pre_mpl(x)
        latent, _ = self.gru(latent)
        latent = self.post_mpl(latent)
        out = self.output(latent)
        out = self.output_activation(out)

        return out


class PhaseRecon(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = Recon()
        self.loss = MSSLoss([1024, 512, 256], alpha=1.0, overlap=0.75)
        self.twopi = nn.Parameter(
            torch.ones(1) * torch.acos(torch.zeros(1)).item() * 4,
            requires_grad=False)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        magnitude_spectrum, audio = batch
        phases = self(magnitude_spectrum) * self.twopi
        phases = phases.permute(0, 2, 1)
        magnitude_spectrum = magnitude_spectrum.permute(0, 2, 1)
        complex_stft = magnitude_spectrum * (torch.cos(phases) + 1j * torch.sin(phases))
        audio_hat = torch.istft(complex_stft,
                                n_fft=1024,
                                hop_length=256,
                                window=torch.hann_window(1024),
                                length=audio.shape[-1],
                                )
        loss = self.loss(audio_hat, audio)
        return loss


def train():
    dataset = AudioData('/home/kureta/Music/violin')
    train_loader = DataLoader(dataset, batch_size=16)
    phase_recon = PhaseRecon()
    trainer = pl.Trainer()
    trainer.fit(phase_recon, train_loader)


if __name__ == '__main__':
    train()
