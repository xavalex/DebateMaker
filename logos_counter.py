import os
import librosa
import numpy as np
import sounddevice as sd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

# Configuration
DATA_DIR = r'G:\My Drive\pers\Python\training_data'
DURATION = 10  # Window size in seconds
SAMPLE_RATE = 22050
BLOCK_SIZE = int(DURATION * SAMPLE_RATE)

def extract_features(audio_data, sr=22050):
    """Extracts MFCC features from a numpy array (audio buffer)."""
    mfccs = librosa.feature.mfcc(y=audio_data.flatten(), sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# --- PHASE 1: TRAINING ---
print("Training models from directory...")
features = []
labels = []

for speaker_dir in os.listdir(DATA_DIR):
    speaker_path = os.path.join(DATA_DIR, speaker_dir)
    if os.path.isdir(speaker_path):
        for file in os.listdir(speaker_path):
            if file.endswith('.wav'):
                audio, _ = librosa.load(os.path.join(speaker_path, file), sr=SAMPLE_RATE)
                features.append(extract_features(audio))
                labels.append(speaker_dir)

# Initialize and fit the LabelEncoder
le = LabelEncoder()
y = le.fit_transform(labels)
X = np.array(features)

# Train the Classifier
clf = svm.SVC(probability=True)
clf.fit(X, y)

# --- PHASE 2: INITIALIZE TALLIES ---
# Now le.classes_ exists and contains the names of your folders
speaker_tallies = {name: 0 for name in le.classes_}
speaker_tallies['Unknown'] = 0

print(f"System Ready! Monitoring for: {list(le.classes_)}")

# --- PHASE 3: CONTINUOUS MONITORING ---
try:
    while True:
        # Record 2 seconds
        recording = sd.rec(BLOCK_SIZE, samplerate=SAMPLE_RATE, channels=1)
        sd.wait() 
        
        # Extract features and predict
        feat = extract_features(recording).reshape(1, -1)
        probs = clf.predict_proba(feat)
        
        # Identify
        if np.max(probs) < 0.6:  # If confidence is low, call it Unknown
            identified_speaker = "Unknown"
        else:
            prediction = clf.predict(feat)
            identified_speaker = le.inverse_transform(prediction)[0]
        
        # Update tally
        speaker_tallies[identified_speaker] += 1
        
        # Display live update
        stats = " | ".join([f"{k}: {v*DURATION}s" for k, v in speaker_tallies.items()])
        print(f"\rCurrent Speaker: {identified_speaker:10} | {stats}", end="")

except KeyboardInterrupt:
    print("\n\nMonitoring stopped. Final Logs:")
    for speaker, count in speaker_tallies.items():
        print(f"{speaker}: {count * DURATION} seconds of talk time.")