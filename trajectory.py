import librosa
import soundfile
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F  # noqa

from config.default import Config
import numpy as np

from crepe.crepe import Crepe
from model.autoencoder.encoder import F0Encoder

default = Config()


# Class to register a hook on the target layer (used to get the output channels of the layer)
class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


# Function to make gradients calculations from the output channels of the target layer
def get_gradients(net_in, net, layer):
    net_in.requires_grad = True
    net.zero_grad()
    hook = Hook(layer)
    net_out = net(net_in)
    loss = hook.output[0].norm()
    loss.backward()
    return net_in.grad.data.squeeze()


def main():
    audio, sr = librosa.load('/home/kureta/Music/gates/31232__thencamenow__metal-gate-03.aiff', sr=16000)
    audio = audio[:len(audio) - (len(audio) % 2048)]
    audio = torch.from_numpy(audio).unsqueeze(0).cuda()

    au_mean = audio.mean(dim=1)
    au_std = audio.std(dim=1)

    net = Crepe().cuda()

    # Function to run the dream.
    def dream(sound, net, layer, iterations, lr):
        sound = sound - au_mean
        sound = sound / au_std

        for i in range(iterations):
            print(f'{i+1}/{iterations}')
            gradients = get_gradients(sound, net, layer)
            sound.data = sound.data + lr * gradients.data

        img_out = sound.detach()
        img_out = img_out * au_std + au_mean
        img_out_np = np.clip(img_out.cpu().numpy(), -1, 1)
        return img_out_np

    layer = list(net.modules())[5]
    for m in net.modules():
        print(m)

    img = dream(audio, net, layer, 20, 10)

    soundfile.write(f'/home/kureta/Music/bok/shit.wav',
                    img[0], sr)

    print('Finished Training')


if __name__ == '__main__':
    main()
