import numpy as np
import streamlit as st

from helper import generated_audio, prepare_inputs, prepare_network

# ======== Sidebar ========
with st.sidebar:
    st.markdown('# Parameters')

    with st.beta_expander('Learning parameters', True):
        lr = st.slider('Learning rate', 0.001, 4., 1.)
        alpha = st.number_input('Content weight', 0., value=1.)
        beta = st.number_input('Style weight', 0., value=1e13)
        max_iter = st.number_input('Maximum learning iterations', 1, value=1000)

    with st.beta_expander('Feature extractor parameters', True):
        kernel_size = st.select_slider('Kernel size', range(3, 34, 2), 11)
        n_features = st.select_slider('Number of features', [2 ** n for n in range(7, 14)], 4096)

    with st.beta_expander('Audio parameters', True):
        sample_rate = st.select_slider('Sample rate', [16000, 22050, 44100, 48000], 44100)
        window_length = st.select_slider('STFT Window Length', [512, 1024, 2048], 2048)
        hop_length = window_length // st.select_slider('Hop factor', [2, 4, 8, 16], 8)

# ======== Main Page ========
st.markdown('# Style transfer for a film')
st.markdown('## Content file options')
content_file = st.file_uploader('Content file', type=['wav', 'mp3', 'flac', 'aiff'])
content = prepare_inputs('content', content_file, sample_rate, window_length, hop_length)

st.markdown('## Style file options')
content_file = st.file_uploader('Style file', type=['wav', 'mp3', 'flac', 'aiff'])
style = prepare_inputs('style', content_file, sample_rate, window_length, hop_length)

if st.button('Start'):
    result = prepare_network(content, style, alpha, beta, lr, max_iter, window_length, hop_length)
    generated_audio(result, sample_rate)
