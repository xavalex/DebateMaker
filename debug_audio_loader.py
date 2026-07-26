import os
from speechbrain.dataio.dataio import read_audio
import torchaudio

# --- Diagnostics and Backend Configuration ---
print("Available torchaudio backends:", torchaudio.list_audio_backends())
try:
    print("Attempting to set backend to FFmpeg...")
    torchaudio.set_audio_backend("ffmpeg")
    print("Successfully set backend to FFmpeg.")
except Exception as e:
    print(f"Could not set FFmpeg backend: {e}")
# -----------------------------------------

# Path to one of the audio files
file_path = os.path.join("training_data", "speaker1", "Francis.wav")

print(f"\nAttempting to load audio file: {file_path}")

if not os.path.exists(file_path):
    print(f"ERROR: File does not exist at '{file_path}'")
else:
    try:
        # Call read_audio but don't unpack the result
        audio_data = read_audio(file_path)
        
        print("\n--- DEBUG ---")
        print(f"Successfully loaded audio data without unpacking.")
        print(f"Type of returned data: {type(audio_data)}")
        
        # If it's a tuple or list, check its length
        if isinstance(audio_data, (tuple, list)):
            print(f"It is a tuple/list with {len(audio_data)} elements.")
            for i, item in enumerate(audio_data):
                print(f"  - Element {i}: type={type(item)}")
                if hasattr(item, 'shape'):
                    print(f"    - Shape: {item.shape}")
                else:
                    print(f"    - Value: {item}")
        else:
            print("The returned data is not a tuple or list.")
            print(f"Data: {audio_data}")
            
    except Exception as e:
        print(f"\nAn error occurred during audio loading: {e}")
