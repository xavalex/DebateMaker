import numpy as np
import sounddevice as sd
import librosa
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_DIR = r'G:\My Drive\pers\Python\training_data'
DURATION = 10  # Window size in seconds
SAMPLE_RATE = 22050
BLOCK_SIZE = int(DURATION * SAMPLE_RATE)

# --- UPDATED TRAINING SECTION ---
# You MUST use a scaler during training if you want live recognition to work
scaler = StandardScaler()

def get_trained_classifier(features, labels):
    X = np.array(features)
    X_scaled = scaler.fit_transform(X) # Save this scaling logic
    clf = svm.SVC(probability=True, kernel='rbf')
    clf.fit(X_scaled, labels)
    return clf

# --- UPDATED LIVE FEATURE EXTRACTION ---
def extract_live_features(audio_buffer, sr=22050):
    # Ensure audio is float32 and normalized like librosa.load
    audio = audio_buffer.flatten().astype(np.float32)
    
    # Simple Noise Gate: Only process if there's actual sound
    if np.max(np.abs(audio)) < 0.02: 
        return None
        
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # IMPROVEMENT: Instead of just mean, let's use mean of non-silent frames
    # This makes the 10-second window much more robust
    mfcc_mean = np.mean(mfccs.T, axis=0)
    return mfcc_mean

# --- THE LOOP ---
try:
    while True:
        print(f"\nRecording {DURATION}s window...")
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        
        raw_features = extract_live_features(recording)
        
        if raw_features is not None:
            # IMPORTANT: Scale the live features using the SAME scaler from training
            scaled_features = scaler.transform(raw_features.reshape(1, -1))
            
            probs = clf.predict_proba(scaled_features)
            max_p = np.max(probs)
            
            if max_p > 0.5: # 50% confidence threshold
                prediction = clf.predict(scaled_features)
                speaker = le.inverse_transform(prediction)[0]
                speaker_tallies[speaker] += 1
                print(f"Detected: {speaker} ({max_p:.2%})")
            else:
                print(f"Unknown (Confidence too low: {max_p:.2%})")
        else:
            print("Silence detected, skipping...")

except KeyboardInterrupt:
    print("Stopped.")