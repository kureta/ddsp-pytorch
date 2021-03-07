import pytorch_lightning as pl
import soundfile
import torch
from torch.utils.data import DataLoader

from config.default import Config
from dataset.audio_dataset import AudioData
from loss.mss_loss import MSSLoss
from model.autoencoder.autoencoder import AutoEncoder

default = Config()


class Zak(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoEncoder()
        self.loss = MSSLoss([2048, 1024, 512])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, x, batch_idx):
        x_hat = self.model(x)
        loss = self.loss(x_hat, x)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, x, batch_idx):
        x_hat = self.model(x)
        for i, x in enumerate(x_hat):
            soundfile.write(f'/home/kureta/Music/junk/shit/{batch_idx}-{i}.wav',
                            x.cpu().numpy(), default.sample_rate)


def main():
    dataset = AudioData()
    train_loader = DataLoader(dataset, batch_size=default.batch_size, shuffle=True, num_workers=4)
    model = Zak()
    trainer = pl.Trainer(gpus=1, limit_val_batches=0.01, precision=16)
    trainer.fit(model, train_loader, train_loader)


if __name__ == '__main__':
    main()
