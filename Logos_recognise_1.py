import os
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuration
DATA_DIR = r'G:\My Drive\pers\Python\training_data'
DURATION = 5  # Seconds to record for recognition
SAMPLE_RATE = 22050

def extract_features(file_path):
    """Extracts MFCC features from an audio file."""
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    
    # Resample if necessary
    if sr != SAMPLE_RATE:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=60)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    features = np.concatenate((np.mean(mfccs.T, axis=0), 
                               np.mean(delta_mfccs.T, axis=0), 
                               np.mean(delta2_mfccs.T, axis=0)))
    return features

## 1. TRAINING PHASE
print("Training models from directory...")
features = []
labels = []

for speaker_dir in os.listdir(DATA_DIR):
    speaker_path = os.path.join(DATA_DIR, speaker_dir)
    if os.path.isdir(speaker_path):
        for file in os.listdir(speaker_path):
            if file.endswith('.wav'):
                file_path = os.path.join(speaker_path, file)
                features.append(extract_features(file_path))
                labels.append(speaker_dir)

# Encode labels (names) into numbers
le = LabelEncoder()
y = le.fit_transform(labels)
X = np.array(features)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the Classifier
clf = svm.SVC(C=10.0, gamma='auto', probability=True)
clf.fit(X, y)
print("System Ready!")

## 2. RECOGNITION PHASE
def recognize_speaker():
    print(f"\n[INVITATION]: Please speak for {DURATION} seconds...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    
    # Save temp file to process
    temp_file = "temp_request.wav"
    write(temp_file, SAMPLE_RATE, recording)
    
    # Identify
    feat = extract_features(temp_file).reshape(1, -1)
    feat = scaler.transform(feat) # Apply the same scaling
    probabilities = clf.predict_proba(feat)[0]

    print("\n--- Speaker Probabilities ---")
    for i, speaker in enumerate(le.classes_):
        print(f"{speaker}: {probabilities[i] * 100:.2f}%")
    
    prediction_idx = np.argmax(probabilities)
    speaker_name = le.inverse_transform([prediction_idx])[0]
    confidence = probabilities[prediction_idx]
    
    print(f"\nI recognize you! You are: {speaker_name} with {confidence * 100:.2f}% confidence.")

if __name__ == "__main__":
    while True:
        recognize_speaker()
        if input("Recognize another? (y/n): ").lower() != 'y':
            break