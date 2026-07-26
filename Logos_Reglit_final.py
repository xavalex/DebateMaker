import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib

# 1. Load the specific file from your path
path = r"G:\My Drive\pers\Python\training_data\Francis\Francis_straight.wav"
y, sr = librosa.load(path, sr=None)

# 2. Pre-processing: Trim and Pre-emphasis
y_trimmed, _ = librosa.effects.trim(y, top_db=20)
y_pre = librosa.effects.preemphasis(y_trimmed, coef=0.97)

# 3. Feature Extraction (20 MFCCs + Delta + Delta-Delta)
mfccs = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=20)
delta = librosa.feature.delta(mfccs)
delta2 = librosa.feature.delta(mfccs, order=2)

# Transform to (n_frames, 60)
features = np.concatenate((mfccs, delta, delta2), axis=0).T

# 4. Feature Normalization (CMVN)
features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

# 5. Dynamic Component Selection
n_frames = features.shape[0]
n_components = max(1, min(16, n_frames // 10))

print(f"Training Francis: {n_frames} frames, {n_components} components.")

# 6. Fit the GMM with Regularization
gmm = GaussianMixture(
    n_components=n_components, 
    covariance_type='diag', 
    reg_covar=1e-5,        # Added regularization to prevent singular matrices
    max_iter=200, 
    random_state=42
)
gmm.fit(features)

# 7. Save the model
output_path = r"G:\My Drive\pers\Python\training_data\Francis\Francis_profile.pkl"
joblib.dump(gmm, output_path)

print(f"Model successfully saved with reg_covar=1e-5.")