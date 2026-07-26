import os
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress FutureWarning from librosa
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa.core.audio')

# Configuration
DATA_DIR = r'G:\My Drive\pers\Python\training_data'
DURATION = 5  # Seconds to record for recognition
SAMPLE_RATE = 22050

def extract_features(file_path):
    """Extracts MFCC features from an audio file."""
    try:
        # Use soundfile engine explicitly
        audio, sr = librosa.load(file_path, sr=None, mono=True, res_type='kaiser_fast')
    except Exception as e:
        print(f"Error loading audio with default engine: {e}")
        # Fallback to audioread if soundfile fails
        audio, sr = librosa.load(file_path, sr=None, mono=True, res_type='kaiser_fast')
    
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

# Train the Classifier
models = {}
for speaker in le.classes_:
    speaker_indices = [i for i, s in enumerate(labels) if s == speaker]
    speaker_features = X[speaker_indices]
    
    gmm = GaussianMixture(n_components=16, covariance_type='diag', n_init=3)
    gmm.fit(speaker_features)
    models[speaker] = gmm

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
    
    scores = {}
    for speaker, model in models.items():
        scores[speaker] = model.score(feat)
        
    prediction = max(scores, key=scores.get)
    
    print(f"\nPrediction: {prediction}")
    
    # Show scores for all speakers
    print("--- All Scores (Log-Likelihood) ---")
    for speaker, score in scores.items():
        print(f"{speaker}: {score:.2f}")

if __name__ == "__main__":
    recognize_speaker()
    while True:
        if input("Recognize another? (y/n): ").lower() != 'y':
            break