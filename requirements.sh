#!/usr/bin/env sh

pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip

pip install JACK-Client librosa matplotlib numpy python-osc SoundFile streamlit jupyter
