import librosa
import numpy as np
import joblib
import sounddevice as sd

def identify_with_guards(y, sr, profiles):
    # 1. SILENCE DETECTION (The Easy One)
    rms = np.sqrt(np.mean(y**2))
    if rms < 0.008:
        return "Silence", 0.0

    # 2. VOICE FINGERPRINTING (Feature Extraction)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_pre = librosa.effects.preemphasis(y_trimmed)
    mfccs = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    features = np.concatenate((mfccs, delta, delta2), axis=0).T
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

    # Calculate LLR for all enrolled models
    scores = {}
    for name, data in profiles.items():
        scores[name] = data['model'].score(features)

    avg_score_across_models = np.mean(list(scores.values()))
    
    # 3. THREE GUARDS
    candidates = []
    for name, score in scores.items():
        profile = profiles[name]
        
        # Guard 0: Floor Check
        g0 = score > profile['floor_threshold']
        
        # Guard 1: Overfit Check (+12 units limit)
        g1 = score < (profile['avg_train_score'] + 12.0)
        
        # Guard 2: Stand-out Check (+5 above average)
        standout_margin = score - avg_score_across_models
        g2 = standout_margin > 5.0
        
        if g0 and g1 and g2:
            candidates.append({
                "name": name,
                "score": score,
                "margin": standout_margin
            })

    # 4. PICKING A WINNER
    if not candidates:
        return "Unknown", 0.0

    # Winner is the one with the largest stand-out margin
    winner = max(candidates, key=lambda x: x['margin'])
    
    # Confidence: Gap between top two margins / 10
    if len(candidates) > 1:
        all_margins = sorted([c['margin'] for c in candidates], reverse=True)
        confidence = (all_margins[0] - all_margins[1]) / 10.0
    else:
        confidence = winner['margin'] / 10.0

    return winner['name'], confidence

# --- Main App Loop ---
# profiles = {"Xavier": joblib.load("Xavier_profile.pkl"), "Francis": ...}