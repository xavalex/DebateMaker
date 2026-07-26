import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
import os

# 1. Load the specific file from your G Drive path
path = r"G:\My Drive\pers\Python\training_data\Xavier\Xavier_20260113_013856.wav"
y, sr = librosa.load(path, sr=None)

# 2. Pre-processing: Trim Silence and Apply Pre-emphasis
y_trimmed, _ = librosa.effects.trim(y, top_db=20)
y_pre = librosa.effects.preemphasis(y_trimmed, coef=0.97)

# 3. Feature Extraction (20 MFCCs + Delta + Delta-Delta)
mfccs = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=20)
delta = librosa.feature.delta(mfccs)
delta2 = librosa.feature.delta(mfccs, order=2)

# Stack to (n_frames, 60)
features = np.concatenate((mfccs, delta, delta2), axis=0).T

# 4. Feature Normalization (CMVN)
# This centers the features to make the model more robust to recording conditions
features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

# 5. Dynamic Component Selection
n_frames = features.shape[0]
n_components = max(1, min(16, n_frames // 10))

print(f"Training profile for Xavier...")
print(f"Frames: {n_frames} | Components: {n_components}")

# 6. Fit the GMM
gmm = GaussianMixture(
    n_components=n_components, 
    covariance_type='diag', 
    max_iter=200, 
    random_state=42
)
gmm.fit(features)

# 7. Save the model to the same directory
output_path = r"G:\My Drive\pers\Python\training_data\Xavier\Xavier_profile.pkl"
joblib.dump(gmm, output_path)

print(f"Model saved successfully to: {output_path}")