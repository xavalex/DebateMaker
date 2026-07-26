import os
from pydub import AudioSegment

def standardize_wav_files(data_folder):
    """
    Finds all .wav files in the subdirectories of a data folder and
    converts them to a standard format (16-bit PCM, 16kHz, mono).
    """
    print(f"Scanning for WAV files in '{data_folder}'...")
    found_files = False
    
    # Iterate through each speaker's directory
    for speaker_dir in os.listdir(data_folder):
        speaker_path = os.path.join(data_folder, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue

        # Find all .wav files in the speaker's directory
        for filename in os.listdir(speaker_path):
            if filename.lower().endswith(".wav"):
                found_files = True
                file_path = os.path.join(speaker_path, filename)
                print(f"  - Standardizing '{file_path}'...")
                
                try:
                    # Load the audio file
                    audio = AudioSegment.from_wav(file_path)
                    
                    # 1. Set to mono
                    audio = audio.set_channels(1)
                    
                    # 2. Set sample rate to 16kHz
                    audio = audio.set_frame_rate(16000)
                    
                    # 3. Export in standard 16-bit PCM format, overwriting the original
                    # The default for .export(format='wav') is 16-bit PCM.
                    audio.export(file_path, format="wav")
                    
                    print(f"    -> Success: File converted to 16kHz mono, 16-bit PCM.")
                
                except Exception as e:
                    print(f"    -> ERROR: Could not process {file_path}. Reason: {e}")

    if not found_files:
        print("\nNo .wav files were found to standardize.")
    else:
        print("\nAudio standardization complete.")

if __name__ == "__main__":
    # The main folder containing speaker subdirectories
    training_data_folder = "training_data"
    
    if not os.path.isdir(training_data_folder):
        print(f"Error: The directory '{training_data_folder}' was not found.")
    else:
        standardize_wav_files(training_data_folder)
