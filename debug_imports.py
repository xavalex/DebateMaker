print("Starting import debug...")
import pyaudio
print("pyaudio imported successfully.")
import torch
print("torch imported successfully.")
import torchaudio
print("torchaudio imported successfully.")
print("Importing speechbrain...")
from speechbrain.inference.classifiers import EncoderClassifier
print("speechbrain.inference.classifiers imported successfully.")
from speechbrain.dataio.dataio import read_audio
print("speechbrain.dataio.dataio imported successfully.")
import numpy as np
print("numpy imported successfully.")
print("All imports successful.")

# --- Settings ---
MODEL_PATH = "speechbrain/spkrec-ecapa-voxceleb"
SAVED_EMBEDDINGS_PATH = "speaker_model_sb/speaker_embeddings.pt"
CHUNK = 1024 * 4 # Amount of audio data to read at a time
FORMAT = pyaudio.paInt16 # Audio format
CHANNELS = 1 # Mono audio
RATE = 44100 # Microphone's native sample rate
TARGET_RATE = 16000 # Model's required sample rate
RECORD_SECONDS = 2 # Duration of audio to process for each prediction
SIMILARITY_THRESHOLD = 0.60 # Minimum similarity to consider a match

# --- Initialization ---
resampler = torchaudio.transforms.Resample(orig_freq=RATE, new_freq=TARGET_RATE)

print("\nLoading model and speaker embeddings...")
try:
    classifier = EncoderClassifier.from_hparams(source=MODEL_PATH)
    loaded_embeddings = torch.load(SAVED_EMBEDDINGS_PATH)
    print("...done.")
except Exception as e:
    print(f"Error loading model or embeddings: {e}")
    print("Please make sure 'speechbrain_training.py' has been run successfully.")
    exit()

p = pyaudio.PyAudio()

# --- Main Loop ---
print("\nOpening microphone stream...")
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening... Press Ctrl+C to stop.")

try:
    # while True:
        print("\nDEBUG: Top of loop.")
        frames = []
        # Record a chunk of audio
        print("DEBUG: Recording audio...")
        # for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        #    data = stream.read(CHUNK)
        #    frames.append(data)
        print("DEBUG: Finished recording.")
        
        

except KeyboardInterrupt:
    print("\nStopping.")
finally:
    # --- Cleanup ---
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Stream closed.")
