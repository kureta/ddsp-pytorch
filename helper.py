import os

import librosa
import numpy as np
import soundfile
import streamlit as st
import torch
import torchaudio
from torch import nn

from style_transfer import ContentLoss, FeatureExtractor, StyleLoss


def normalize_audio(x):
    # Remove DC Offset
    x = x - x.mean()
    # Normalize
    x = x / np.max(np.abs(x))

    return x


def prepare_spectra(file_path, sample_rate, win_length, hop_length):
    audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    audio = normalize_audio(audio)
    complex_spectrum = librosa.stft(audio, n_fft=win_length, hop_length=hop_length)
    mag = np.abs(complex_spectrum)
    db = np.log1p(mag)

    # (channels, frames)
    return db, audio


def generated_audio(audio, sample_rate):
    soundfile.write('tmp.wav', audio, sample_rate)
    st.audio('tmp.wav')
    os.remove('tmp.wav')


def prepare_inputs(name, audio_file, sample_rate, window_length, hop_length):
    if audio_file is not None:
        st.audio(audio_file)
        content, content_audio = prepare_spectra(audio_file, sample_rate, window_length, hop_length)
        content_length = len(content_audio)
        content_duration = content_length / sample_rate

        content_start_time = st.slider(f'{name} start', 0., content_duration, 0.)
        content_end_time = st.slider(f'{name} end', content_start_time, content_duration, content_duration)

        content_start_frame = librosa.time_to_frames(content_start_time, sample_rate, hop_length)
        content_end_frame = librosa.time_to_frames(content_end_time, sample_rate, hop_length)

        content_start = int(content_start_time * sample_rate)
        content_end = int(content_end_time * sample_rate)

        st.markdown(f'Trimmed {name} file')
        generated_audio(content_audio[content_start:content_end], sample_rate)

        content_trimmed = content[:, content_start_frame:content_end_frame]
        content_img = np.flip(content_trimmed, axis=0)
        content_img -= content_img.min()
        content_img /= content_img.max()
        st.image(content_img, f'Trimmed {name} spectrogram', use_column_width=True)

        return content_trimmed


def prepare_network(content, style, alpha, beta, lr, max_iter, win_length, hop_length):
    n_channels = content.shape[0]

    elem_mean = np.mean(content)
    elem_std = np.std(content)

    content = torch.from_numpy(np.ascontiguousarray(content)).unsqueeze(0).cuda()
    style = torch.from_numpy(np.ascontiguousarray(style)).unsqueeze(0).cuda()

    net = nn.Sequential(FeatureExtractor(n_channels, 4096, 17).cuda())

    with torch.no_grad():
        content_features = net(content)
        style_features = net(style)

    content_loss = ContentLoss(content_features)
    net.add_module('content_loss', content_loss)
    style_loss = StyleLoss(style_features)
    net.add_module('style_loss', style_loss)

    optimizer = torch.optim.LBFGS([content.requires_grad_()], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        net(content)
        loss = beta * net.style_loss.loss + alpha * net.content_loss.loss
        loss.backward()

        return loss

    optimizer.step(closure)

    with torch.no_grad():
        content = content * elem_std + elem_mean
        # content_length = librosa.frames_to_samples(content.shape[1], hop_length)

        result = torch.exp(content) - 1
        result = torchaudio.functional.griffinlim(result,
                                                  window=torch.hann_window(win_length, True).cuda(),
                                                  n_fft=win_length,
                                                  hop_length=hop_length,
                                                  win_length=win_length,
                                                  power=1, n_iter=1000, momentum=0.99,
                                                  length=None,
                                                  rand_init=True)
        result = result.cpu().numpy()[0]
        result = normalize_audio(result)

        return result
