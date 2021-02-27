import glob
import os

import pytorch_lightning as pl
import soundfile
import torch
import torch.fft
import torch.nn.functional as F  # noqa
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from loss.mss_loss import MSSLoss


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
        window = torch.hann_window(1024)
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
            spec = torch.stft(y, n_fft=1024, hop_length=256, window=window)
            spec = spec.pow(2.).sum(-1).pow(0.5)
            # normalize to [0, 1]
            spec /= 512
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


class Highs(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(513, 512, 3, 1, 0, 0)
        self.fire1 = nn.Conv1d(512, 256, 1, 1)
        self.unfire1 = nn.Conv1d(256, 512, 1, 1)
        self.conv2 = nn.ConvTranspose1d(512, 256, 7, 4, 0, 0)
        self.fire2 = nn.Conv1d(256, 128, 1, 1)
        self.unfire2 = nn.Conv1d(128, 256, 1, 1)
        self.conv3 = nn.ConvTranspose1d(256, 128, 7, 4, 0, 0)
        self.fire3 = nn.Conv1d(128, 64, 1, 1)
        self.unfire3 = nn.Conv1d(64, 128, 1, 1)
        self.conv4 = nn.ConvTranspose1d(128, 64, 7, 4, 0, 0)
        self.fire4 = nn.Conv1d(64, 32, 1, 1)
        self.unfire4 = nn.Conv1d(32, 64, 1, 1)
        self.conv5 = nn.ConvTranspose1d(64, 1, 8, 4, 0, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.elu(out)

        burnt = F.elu(self.fire1(out))
        burnt = F.elu(self.unfire1(burnt))
        out = out + burnt

        out = self.conv2(out)
        out = F.elu(out)

        burnt = F.elu(self.fire2(out))
        burnt = F.elu(self.unfire2(burnt))
        out = out + burnt

        out = self.conv3(out)
        out = F.elu(out)

        burnt = F.elu(self.fire3(out))
        burnt = F.elu(self.unfire3(burnt))
        out = out + burnt

        out = self.conv4(out)
        out = F.elu(out)

        burnt = F.elu(self.fire4(out))
        burnt = F.elu(self.unfire4(burnt))
        out = out + burnt

        out = self.conv5(out)
        out = torch.tanh(out)

        return out


class Lows(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(513, 512, 3, 1, 0, 0)
        self.fire1 = nn.Conv1d(512, 256, 1, 1)
        self.unfire1 = nn.Conv1d(256, 512, 1, 1)
        self.conv2 = nn.ConvTranspose1d(512, 256, 3, 2, 0, 0)
        self.fire2 = nn.Conv1d(256, 128, 1, 1)
        self.unfire2 = nn.Conv1d(128, 256, 1, 1)
        self.conv3 = nn.ConvTranspose1d(256, 128, 15, 8, 0, 0)
        self.fire3 = nn.Conv1d(128, 64, 1, 1)
        self.unfire3 = nn.Conv1d(64, 128, 1, 1)
        self.conv4 = nn.ConvTranspose1d(128, 1, 16, 8, 0, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.elu(out)

        burnt = F.elu(self.fire1(out))
        burnt = F.elu(self.unfire1(burnt))
        out = out + burnt

        out = self.conv2(out)
        out = F.elu(out)

        burnt = F.elu(self.fire2(out))
        burnt = F.elu(self.unfire2(burnt))
        out = out + burnt

        out = self.conv3(out)
        out = F.elu(out)

        burnt = F.elu(self.fire3(out))
        burnt = F.elu(self.unfire3(burnt))
        out = out + burnt

        out = self.conv4(out)
        out = torch.tanh(out)

        return out


class Bass(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(513, 512, 3, 1, 0, 0)
        self.fire1 = nn.Conv1d(512, 256, 1, 1)
        self.unfire1 = nn.Conv1d(256, 512, 1, 1)
        self.conv2 = nn.ConvTranspose1d(512, 256, 15, 8, 0, 0)
        self.fire2 = nn.Conv1d(256, 128, 1, 1)
        self.unfire2 = nn.Conv1d(128, 256, 1, 1)
        self.conv3 = nn.ConvTranspose1d(256, 1, 16, 8, 0, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.elu(out)

        burnt = F.elu(self.fire1(out))
        burnt = F.elu(self.unfire1(burnt))
        out = out + burnt

        out = self.conv2(out)
        out = F.elu(out)

        burnt = F.elu(self.fire2(out))
        burnt = F.elu(self.unfire2(burnt))
        out = out + burnt

        out = self.conv3(out)
        out = torch.tanh(out)

        return out


class Recon(nn.Module):
    def __init__(self):
        super().__init__()
        self.lows = Lows()
        self.highs = Highs()
        self.bass = Bass()
        self.resample = nn.Upsample(scale_factor=2)
        self.resample_bass = nn.Upsample(scale_factor=4)

    def forward(self, x):
        bass = self.bass(x)
        bass = self.resample_bass(bass)
        lows = self.lows(x)
        lows = self.resample(lows)
        highs = self.highs(x)
        out = bass + lows + highs

        return out


class ISTFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.ConvTranspose1d(513, 512, 3, 1)
        self.first_elu = nn.ELU()
        self.layers = nn.ModuleList(
            [nn.ConvTranspose1d(512 // (2 ** n), 512 // (2 ** (n + 1)), 3, 2) for n in range(7)])
        self.patches = nn.ModuleList([nn.Conv1d(512 // (2 ** n), 512 // (2 ** (n + 1)), 1, 1) for n in range(7)])
        self.last_layer = nn.ConvTranspose1d(512 // (2 ** 7), 1, 4, 2)

    def forward(self, x):
        outs = []
        out = self.first_elu(self.first_layer(x))
        outs.append(out)
        for layer in self.layers:
            out = torch.sin(layer(out))
            res = F.interpolate(outs[-1], size=out.size(-1))
            res = self.patches[len(outs) - 1](res)
            out = out + res
            outs.append(out)
        out = torch.sin(self.last_layer(out))
        return out


class PhaseRecon(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = ISTFT()
        self.loss = MSSLoss([1024, 512, 256], alpha=1.0, overlap=0.75)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        magnitude_spectrum, audio = batch
        magnitude_spectrum = magnitude_spectrum.permute(0, 2, 1)
        audio_hat = self(magnitude_spectrum)
        audio_hat = audio_hat.squeeze(1)
        loss = self.loss(audio_hat, audio)
        return loss

    def validation_step(self, batch, batch_idx):
        magnitude_spectrum, audio = batch
        magnitude_spectrum = magnitude_spectrum.permute(0, 2, 1)
        audio_hat = self(magnitude_spectrum)
        audio_hat = audio_hat.squeeze(1)
        for idx, audio in enumerate(audio_hat):
            soundfile.write(f'/home/kureta/Music/junk/shit/{batch_idx}-{idx}.wav', audio.cpu().numpy(), 44100, 'PCM_24')


def train():
    dataset = AudioData('/home/kureta/Music/violin')
    train_loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
    phase_recon = PhaseRecon()
    trainer = pl.Trainer(gpus=1, limit_val_batches=0.01,
                         resume_from_checkpoint='lightning_logs/version_23/checkpoints/epoch=170-step=72060.ckpt')
    trainer.fit(phase_recon, train_loader, train_loader)


if __name__ == '__main__':
    train()
