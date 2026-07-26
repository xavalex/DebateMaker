import os
import librosa
import numpy as np
import sounddevice as sd
from sklearn import svm
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configuration
DATA_DIR = r'G:\My Drive\pers\Python\training_data'
SAMPLE_RATE = 22050
DURATION = 5  # We'll stick to 5s for now
N_MFCC = 80    # Increased from 40 for finer vowel detail
N_MELS = 128   # Number of Mel bands to distribute filters across

def extract_detailed_features(audio, sr=22050):
    # 1. Pre-emphasis filter: boosts high frequencies to highlight vowel formants
    audio_emphasized = librosa.effects.preemphasis(audio)
    
    # 2. Extract Mel Spectrogram with 128 bands
    S = librosa.feature.melspectrogram(y=audio_emphasized, sr=sr, n_mels=N_MELS)
    
    # 3. Convert to log scale (decibels)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # 4. Extract 80 MFCCs from the log-power Mel spectrogram
    mfccs = librosa.feature.mfcc(S=log_S, n_mfcc=N_MFCC)
    
    # 5. Return the mean across time
    return np.mean(mfccs.T, axis=0)

# --- TRAINING PHASE ---
print("Training with 80 Mel Filters...")
features = []
labels = []

for speaker_dir in os.listdir(DATA_DIR):
    speaker_path = os.path.join(DATA_DIR, speaker_dir)
    if os.path.isdir(speaker_path):
        for file in os.listdir(speaker_path):
            if file.endswith('.wav'):
                audio, _ = librosa.load(os.path.join(speaker_path, file), sr=SAMPLE_RATE)
                features.append(extract_detailed_features(audio))
                labels.append(speaker_dir)

le = LabelEncoder()
y = le.fit_transform(labels)
X = np.array(features)

# Use StandardScaler: Crucial when using 80 filters to normalize the range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = svm.SVC(probability=True, kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_scaled, y)
print("Model Trained.")

# --- LIVE RECOGNITION LOOP ---
speaker_tallies = {name: 0 for name in le.classes_}

try:
    while True:
        print(f"\nListening for {DURATION}s...")
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        
        # Flatten and extract
        live_audio = recording.flatten().astype(np.float32)
        live_feat = extract_detailed_features(live_audio).reshape(1, -1)
        
        # Scale based on training stats
        live_feat_scaled = scaler.transform(live_feat)
        
        probs = clf.predict_proba(live_feat_scaled)
        max_p = np.max(probs)
        
        if max_p > 0.45:  # Adjusted threshold
            prediction = clf.predict(live_feat_scaled)
            speaker = le.inverse_transform(prediction)[0]
            speaker_tallies[speaker] += 1
            print(f"Match: {speaker} ({max_p:.2%})")
        else:
            print(f"Unknown Speaker (Confidence: {max_p:.2%})")

except KeyboardInterrupt:
    print("\nStopped.")