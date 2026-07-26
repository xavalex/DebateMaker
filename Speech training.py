import pyaudio
import wave
import numpy as np
import os
from pyAudioAnalysis import audioBasicIO as aIO
# from pyAudioAnalysis import audioFeatureExtraction as aF
#from pyAudioAnalysis import pyAudioAnalysis

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Sample rate in Hz
CHUNK = 1024
TRAINING_DURATION = 10  # in seconds
OUTPUT_DIR = "training_data"

def record_audio(duration, filename):
    """Records audio from the microphone for a specified duration and saves it."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print(f"Recording {filename} for {duration} seconds...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    print(f"Finished recording {filename}.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio saved to {filename}")

def extract_features(filename):
    """Extracts features from the WAV file and returns them."""
    [Fs, x] = aIO.read_audio_file(filename)
    F = pyAudioAnalysis.pyAudioAnalysis.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    return F.T

def main():
    """Main function to run the training process."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    speaker_names = []
    for i in range(2):
        name = input(f"Enter name for speaker {i+1}: ")
        speaker_names.append(name)
        speaker_dir = os.path.join(OUTPUT_DIR, name)
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)

        # Record and save audio for this speaker
        wav_filename = os.path.join(speaker_dir, f"{name}_sample.wav")
        record_audio(TRAINING_DURATION, wav_filename)

    print("\nTraining audio files have been saved.")
    print("Now, you need to train the classification model.")
    print("Run the following command in your terminal:")
    print(f"python audioAnalysis.py trainClassifier -i {OUTPUT_DIR}/ --method svm --modelName speaker_model")

if __name__ == "__main__":
    main()