import sys
import json
import os
import joblib
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# When there is only 1 enrolled speaker (no cross-scores available),
# fall back to this margin below self_score as the unknown threshold.
SINGLE_SPEAKER_MARGIN = 15.0

# How many log-likelihood units to go below the worst cross-speaker score
# to set the unknown rejection threshold per model.
CROSS_SCORE_MARGIN = 5.0
# Hard floor: unknown_threshold may never sit closer than this many units
# below a speaker's own self_score, keeping natural score variance covered.
MIN_SELF_SCORE_GAP = 18.0


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y, _ = librosa.effects.trim(y)
        y = librosa.effects.preemphasis(y)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
        return features.T
    except Exception as e:
        sys.stderr.write(f"Feature error for {file_path}: {str(e)}\n")
        return None


def main():
    data = json.loads(sys.stdin.read())
    # enrollment_samples: { "0": ["path1", "path2"], "1": ["path3"] }
    enrollment_samples = data['enrollment_samples']
    output_dir = data['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    speaker_features = {}
    all_features = []

    for idx, paths in enrollment_samples.items():
        feats = []
        for path in paths:
            if not os.path.exists(path):
                sys.stderr.write(f"File not found: {path}\n")
                continue
            f = extract_features(path)
            if f is not None and len(f) > 0:
                feats.append(f)
        if feats:
            combined = np.vstack(feats)
            speaker_features[idx] = combined
            all_features.append(combined)

    if not all_features:
        print(json.dumps({"error": "No features could be extracted from enrollment audio"}))
        return

    # Fit a global scaler across all speakers for consistency
    global_data = np.vstack(all_features)
    scaler = StandardScaler()
    scaler.fit(global_data)

    # --- Phase 1: train all GMMs and save preliminary models ---
    trained_models = {}
    trained = []
    for idx, features in speaker_features.items():
        scaled = scaler.transform(features)
        n_components = min(8, len(scaled) // 20)
        if n_components < 1:
            n_components = 1

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='diag',
            n_init=5,
            reg_covar=1e-3,
            max_iter=200,
        )
        gmm.fit(scaled)

        self_score = float(gmm.score(scaled))
        trained_models[idx] = {'gmm': gmm, 'scaler': scaler, 'self_score': self_score}
        trained.append(idx)

    # --- Phase 2: compute cross-scores and set per-model unknown thresholds ---
    for idx, model_data in trained_models.items():
        gmm = model_data['gmm']

        cross_scores = []
        for other_idx, other_features in speaker_features.items():
            if other_idx == idx:
                continue
            other_scaled = scaler.transform(other_features)
            cross_scores.append(float(gmm.score(other_scaled)))

        if cross_scores:
            # Threshold sits CROSS_SCORE_MARGIN below the worst enrolled cross-score.
            # Any voice scoring worse than this on ALL models is considered unknown.
            threshold = min(cross_scores) - CROSS_SCORE_MARGIN
        else:
            # Only one speaker enrolled — fall back to margin below self_score
            threshold = model_data['self_score'] - SINGLE_SPEAKER_MARGIN

        # Hard floor: never sit closer than MIN_SELF_SCORE_GAP below self_score
        floor = model_data['self_score'] - MIN_SELF_SCORE_GAP
        model_data['unknown_threshold'] = min(threshold, floor)

        # Save raw (unscaled) features so session_adapt.py can combine them
        # with new confirmed live-meeting chunks for online adaptation.
        model_data['enrollment_features'] = features

        model_path = os.path.join(output_dir, f"session_{idx}.pkl")
        joblib.dump(model_data, model_path)

        cross_str = ', '.join(f'{s:.2f}' for s in cross_scores) if cross_scores else 'n/a'
        sys.stderr.write(
            f"Trained model for speaker {idx} with {model_data['gmm'].n_components} components, "
            f"self_score={model_data['self_score']:.2f}, "
            f"cross_scores=[{cross_str}], "
            f"unknown_threshold={model_data['unknown_threshold']:.2f}\n"
        )

    print(json.dumps({"ok": True, "trained": trained}))


if __name__ == "__main__":
    main()