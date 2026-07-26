import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from datetime import datetime

# --- Configuration ---
# Make sure these match the settings in your recognition script
DATA_DIR = r'G:\My Drive\pers\Python\training_data'
DURATION = 5  # Seconds to record for each sample
SAMPLE_RATE = 22050

def record_and_save():
    """
    Main function to handle the recording process.
    """
    # Get speaker's name
    speaker_name = input("Enter the speaker's name (or press Enter to quit): ").strip()
    if not speaker_name:
        return False # Signal to exit the loop

    # Sanitize speaker name to create a valid directory name
    sanitized_speaker_name = "".join(c for c in speaker_name if c.isalnum() or c in (' ', '_')).rstrip()
    speaker_dir = os.path.join(DATA_DIR, sanitized_speaker_name)

    # Create directory for the speaker if it doesn't exist
    os.makedirs(speaker_dir, exist_ok=True)

    # Record audio
    print(f"\nGet ready to record for {speaker_name}...")
    print("Recording in 3... 2... 1...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # Generate a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitized_speaker_name}_{timestamp}.wav"
    filepath = os.path.join(speaker_dir, filename)

    # Save the recording as a WAV file
    # Ensure the data is in the correct format (e.g., int16) for WAV
    write(filepath, SAMPLE_RATE, (recording * np.iinfo(np.int16).max).astype(np.int16))
    
    print(f"Successfully saved sample to: {filepath}\n")
    return True # Signal to continue the loop

if __name__ == "__main__":
    print("--- Audio Sample Recorder ---")
    print(f"Samples will be saved in subdirectories of: {DATA_DIR}")
    print(f"Each recording will be {DURATION} seconds long.")
    print("-----------------------------")
    
    while record_and_save():
        pass # Continue looping until the user quits

    print("Exiting recorder.")
