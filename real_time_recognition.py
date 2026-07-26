print("Script starting...")
import pyaudio
import torch
import os
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
import numpy as np
import torchaudio


# --- Settings ---
MODEL_PATH = "speechbrain/spkrec-ecapa-voxceleb"
SAVED_EMBEDDINGS_PATH = "speaker_model_sb/speaker_embeddings.pt"
CHUNK = 1024 * 4 # Amount of audio data to read at a time
FORMAT = pyaudio.paInt16 # Audio format
CHANNELS = 2 # Microphone's native channel count
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
    while True:
        print("\nDEBUG: Top of loop.")
        frames = []
        # Record a chunk of audio
        print("DEBUG: Recording audio...")
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("DEBUG: Finished recording.")

        print("DEBUG: Processing audio...")
        # Convert the recorded data to a torch tensor
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        signal = torch.from_numpy(audio_data).float()

        # Normalize, reshape for stereo, and convert to mono
        signal = signal / 32768.0
        if CHANNELS > 1:
            signal = signal.reshape(-1, CHANNELS)
            signal = torch.mean(signal, dim=1) # Average channels to get mono

        # Resample to the model's expected sample rate
        resampled_signal = resampler(signal.unsqueeze(0)).squeeze()
        print("DEBUG: Finished processing audio.")

        # Get the embedding of the live audio
        print("DEBUG: Encoding audio...")
        live_embedding = classifier.encode_batch(resampled_signal.unsqueeze(0)).squeeze()
        print("DEBUG: Finished encoding.")

        # Compare with enrolled speakers
        print("DEBUG: Comparing embeddings...")
        best_match_speaker = "Unknown"
        highest_similarity = -1.0

        for speaker, enrolled_embedding in loaded_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(live_embedding, enrolled_embedding, dim=0)
            if similarity.item() > highest_similarity:
                highest_similarity = similarity.item()
                best_match_speaker = speaker
        print("DEBUG: Finished comparing.")

        # Print the result
        if highest_similarity > SIMILARITY_THRESHOLD:
            print(f"Prediction: {best_match_speaker} (Similarity: {highest_similarity:.2f})")
        else:
            print(f"Prediction: Unknown (Highest Similarity: {highest_similarity:.2f})")


except KeyboardInterrupt:
    print("\nStopping.")
finally:
    # --- Cleanup ---
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Stream closed.")
