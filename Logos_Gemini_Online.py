import librosa
import numpy as np
from sklearn.mixture import GaussianMixture

# 1. Feature Extraction (The 60 coefficients from previous step)
y, sr = librosa.load('voice_sample.wav', sr=None)
y_trimmed, _ = librosa.effects.trim(y, top_db=20)
y_pre = librosa.effects.preemphasis(y_trimmed)

mfccs = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=20)
delta = librosa.feature.delta(mfccs)
delta2 = librosa.feature.delta(mfccs, order=2)

# Stack to get (60, n_frames) then transpose to (n_frames, 60)
# GMM expects (samples, features)
features_60 = np.concatenate((mfccs, delta, delta2), axis=0).T

# 2. Define the GMM
# n_components=16 as requested
# covariance_type='diag' is standard for MFCCs because they are decorrelated
gmm = GaussianMixture(n_components=16, covariance_type='diag', max_iter=200, random_state=42)

# 3. Fit the model to the speaker's data
gmm.fit(features_60)

print("GMM training complete.")
print(f"Log-likelihood of the data: {gmm.score(features_60)}")