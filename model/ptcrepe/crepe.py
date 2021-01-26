import sys

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torchaudio

from .utils import *


class ConvBlock(nn.Module):
    def __init__(self, f, w, s, in_channels):
        super().__init__()
        p1 = (w - 1) // 2
        p2 = (w - 1) - p1
        self.pad = nn.ZeroPad2d((0, 0, p1, p2))

        self.conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=f, kernel_size=(w, 1), stride=s
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(f)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CREPE(nn.Module):
    def __init__(self, model_capacity="full"):
        super().__init__()

        capacity_multiplier = {"tiny": 4, "small": 8, "medium": 16, "large": 24, "full": 32}[
            model_capacity
        ]
        self.mapping = torch.linspace(0, 7180, 360) + 1997.3794084376191

        self.layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        filters = [1] + filters
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        for i in range(len(self.layers)):
            f, w, s, in_channel = filters[i + 1], widths[i], strides[i], filters[i]
            self.add_module("conv%d" % i, ConvBlock(f, w, s, in_channel))

        self.linear = nn.Linear(64 * capacity_multiplier, 360)
        self.load_weight(model_capacity)
        self.eval()

    def load_weight(self, model_capacity):
        download_weights(model_capacity)
        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "crepe-{}.pth".format(model_capacity)
        self.load_state_dict(torch.load(os.path.join(package_dir, filename)))

    def forward(self, x):
        # x : shape (batch, sample)
        x = x.view(x.shape[0], 1, -1, 1)
        for i in range(len(self.layers)):
            x = self.__getattr__("conv%d" % i)(x)

        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

    def get_activation(self, audio, sr):
        """     
        audio : (N,) or (C, N)
        """

        if sr != 16000:
            rs = torchaudio.transforms.Resample(sr, 16000)
            audio = rs(audio)

        missing = 1024 - audio.shape[1]
        pre = missing // 2
        post = missing - pre
        audio = nn.functional.pad(audio, pad=(pre, post))
        activations = self.forward(audio)

        return activations

    def predict(self, audio, sr):
        activation = self.get_activation(audio, sr)
        frequency = self.to_freq(activation)
        confidence = activation.max(dim=1)[0]
        return frequency, confidence, activation

    def process_file(
            self,
            file,
    ):
        try:
            audio, sr = torchaudio.load(file)  # noqa
        except ValueError:
            print("CREPE-pytorch : Could not read", file, file=sys.stderr)
            return

        with torch.no_grad():
            frequency, confidence, activation = self.predict(
                audio,
                sr,
            )

        return frequency, confidence, activation

    def to_local_average_cents(self, salience, center=None):
        """
        find the weighted average cents near the argmax bin
        """
        self.mapping = self.mapping.to(salience.device)

        if salience.ndim == 1:
            if center is None:
                center = int(torch.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            product_sum = torch.sum(salience * self.mapping[start:end])
            weight_sum = torch.sum(salience)
            return product_sum / weight_sum
        if salience.ndim == 2:
            temp = [self.to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])]
            res = torch.stack(temp)
            return res

        raise Exception("label should be either 1d or 2d Tensor")

    def to_freq(self, activation):
        cents = self.to_local_average_cents(activation)
        frequency = 10 * 2 ** (cents / 1200)
        frequency[torch.isnan(frequency)] = 0
        return frequency
