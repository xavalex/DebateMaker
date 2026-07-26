import time
import joblib
import librosa
import numpy as np
import sounddevice as sd

# --- Configuration ---
MODEL_PATH = r"G:\My Drive\pers\Python\training_data"
SPEAKERS = ["Xavier", "Francis", "Guest"]  # Names of your .pkl files
CHUNK_DURATION = 2  # Seconds per analysis window
FS = 44100         # Sample rate

# Initialize timers
talk_time = {speaker: 0 for speaker in SPEAKERS}
talk_time["Unknown"] = 0
talk_time["Silence"] = 0

def get_features(y, sr):
    """Consistent 60-feature extraction pipeline."""
    y_pre = librosa.effects.preemphasis(y, coef=0.97)
    mfccs = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    features = np.concatenate((mfccs, delta, delta2), axis=0).T
    # CMVN Normalization
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    return features

def identify_chunk(y, sr, profiles):
    # 1. Silence Detection
    rms = np.sqrt(np.mean(y**2))
    if rms < 0.008:
        return "Silence"

    # 2. Feature Extraction
    try:
        features = get_features(y, sr)
    except:
        return "Silence" # Handle empty/tiny buffers

    # 3. Calculate Scores
    scores = {name: p['model'].score(features) for name, p in profiles.items()}
    avg_score = np.mean(list(scores.values()))
    
    # 4. The Three Guards
    candidates = []
    for name, score in scores.items():
        p = profiles[name]
        g0 = score > p['floor_threshold']          # Floor Guard
        g1 = score < (p['avg_train_score'] + 12.0) # Overfit Guard
        g2 = (score - avg_score) > 5.0             # Stand-out Guard
        
        if g0 and g1 and g2:
            candidates.append({"name": name, "margin": score - avg_score})

    if not candidates:
        return "Unknown"

    # Winner has the highest stand-out margin
    winner = max(candidates, key=lambda x: x['margin'])
    return winner['name']

def start_debate(duration_minutes=5):
    # Load Profiles
    print("Initializing Debate System...")
    profiles = {}
    for s in SPEAKERS:
        profiles[s] = joblib.load(f"{MODEL_PATH}\{s}_profile.pkl")

    total_chunks = int((duration_minutes * 60) / CHUNK_DURATION)
    
    print(f"\n--- Debate Started ({duration_minutes} min) ---")
    print("Press Ctrl+C to end early and see stats.\n")

    try:
        for _ in range(total_chunks):
            # Record chunk
            audio = sd.rec(int(CHUNK_DURATION * FS), samplerate=FS, channels=1)
            sd.wait()
            y = audio.flatten()

            # Identify
            speaker = identify_chunk(y, FS, profiles)
            
            # Update Statistics
            talk_time[speaker] += CHUNK_DURATION
            
            # Real-time feedback
            print(f"[Live] Currently Speaking: {speaker.upper()}", end='\r')

    except KeyboardInterrupt:
        print("\n\nDebate interrupted by user.")

    # --- Final Report ---
    print("\n" + "="*30)
    print("      DEBATE STATISTICS      ")
    print("="*30)
    total_active_time = sum(talk_time.values())
    
    for speaker, seconds in talk_time.items():
        percentage = (seconds / total_active_time) * 100
        mins, secs = divmod(seconds, 60)
        print(f"{speaker:10} : {int(mins):02d}m {int(secs):02d}s ({percentage:.1f}%)")
    print("="*30)

if __name__ == "__main__":
    start_debate(duration_minutes=2) # Adjust as needed