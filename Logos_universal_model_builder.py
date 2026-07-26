import os
import librosa
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture

# --- Configuration ---
BASE_DIR = r"G:\My Drive\pers\Python\training_data"
SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.flac')

def get_speaker_features(speaker_folder):
    """Iterates through all audio files in a folder and aggregates features."""
    all_features = []
    
    # List all files in the speaker's directory
    audio_files = [f for f in os.listdir(speaker_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    
    if not audio_files:
        print(f"  [!] No audio files found in {speaker_folder}")
        return None

    for file_name in audio_files:
        file_path = os.path.join(speaker_folder, file_name)
        try:
            # 1. Load
            y, sr = librosa.load(file_path, sr=None)
            
            # 2. Pre-processing: Trim and Pre-emphasis
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            y_pre = librosa.effects.preemphasis(y_trimmed, coef=0.97)

            # 3. Feature Extraction (20 MFCCs + Delta + Delta-Delta)
            mfccs = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=20)
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)

            # Transform to (n_frames, 60)
            file_features = np.concatenate((mfccs, delta, delta2), axis=0).T
            
            # 4. Feature Normalization (CMVN)
            file_features = (file_features - np.mean(file_features, axis=0)) / (np.std(file_features, axis=0) + 1e-8)
            
            all_features.append(file_features)
            print(f"  [+] Processed: {file_name}")
            
        except Exception as e:
            print(f"  [!] Error processing {file_name}: {e}")

    # Stack all features from all files into one large matrix
    return np.vstack(all_features) if all_features else None

def run_universal_training():
    # Identify subdirectories (each directory is a speaker)
    speakers = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    for speaker in speakers:
        print(f"\n>>> Training Model for: {speaker}")
        speaker_path = os.path.join(BASE_DIR, speaker)
        
        # Aggregate features from all files in the folder
        features = get_speaker_features(speaker_path)
        
        if features is not None:
            n_frames = features.shape[0]
            # 5. Dynamic Component Selection (Standard logic)
            n_components = max(1, min(16, n_frames // 10))
            
            print(f"  [*] Training GMM: {n_frames} total frames, {n_components} components.")

            # 6. Fit the GMM
            gmm = GaussianMixture(
                n_components=n_components, 
                covariance_type='diag', 
                reg_covar=1e-5, 
                max_iter=200, 
                random_state=42
            )
            gmm.fit(features)

            # 7. Save the model in the same folder
            output_path = os.path.join(speaker_path, f"{speaker}_profile.pkl")
            joblib.dump(gmm, output_path)
            print(f"  [SUCCESS] Model saved to: {output_path}")

if __name__ == "__main__":
    run_universal_training()