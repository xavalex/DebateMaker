print("Script starting...")
import pyaudio
import torch
print(torch.__version__)
import os
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
import numpy as np
import torchaudio
import warnings

print("\imports successful")


