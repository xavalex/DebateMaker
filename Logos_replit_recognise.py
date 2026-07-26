import os
import joblib
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# --- Configuration ---
MODEL_DIR = r"G:\My Drive\pers\Python\training_data"
MODELS = {
    "Xavier": os.path.join(MODEL_DIR, "Xavier", "Xavier_profile.pkl"),
    "Francis": os.path.join(MODEL_DIR, "Francis", "Francis_profile.pkl")
}
DURATION = 5  # seconds to record
FS = 44100    # sample rate

def extract_60_features(y, sr):
    """Consistent pipeline with training: Trim -> Pre-emphasis -> 20 MFCC + D + DD -> CMVN"""
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_pre = librosa.effects.preemphasis(y_trimmed, coef=0.97)
    
    mfccs = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    
    features = np.concatenate((mfccs, delta, delta2), axis=0).T
    # Apply CMVN (Crucial for matching the training data distribution)
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    return features

def main():
    # 1. Load both models
    print("Loading models...")
    gmm_models = {name: joblib.load(path) for name, path in MODELS.items()}

    # 2. Record voice
    print(f"\n[Ready] Please speak for {DURATION} seconds...")
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished. Identifying...")

    # Convert to 1D array for librosa
    y = recording.flatten()
    # y, sr = librosa.load(r"G:\My Drive\pers\Python\training_data\Xavier\Xavier_20260113_013856.wav", sr=None)

    # 3. Process features
    try:
        features = extract_60_features(y, FS)
        
        # 4. Score against models
        scores = {name: gmm.score(features) for name, gmm in gmm_models.items()}
        print(scores)
        
        # 5. Determine Winner
        winner = max(scores, key=scores.get)
        confidence = scores[winner]

        print("-" * 30)
        print(f"IDENTIFIED SPEAKER: {winner.upper()}")
        print(f"Log-Likelihood: {confidence:.2f}")
        print("-" * 30)
        
        for name, score in scores.items():
            print(f"{name}: {score:.2f}")

    except Exception as e:
        print(f"Error during identification: {e}")

if __name__ == "__main__":
    main()