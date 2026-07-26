import google.generativeai as genai
import os

# Configure your API key
genai.configure(api_key="AIzaSyB3oXNknwh-gLzscgykfTXIYoGp1Jb7DWs")

# Configuration
DATA_DIR = r'G:\My Drive\pers\Python\training_data'
DURATION = 5  # Seconds to record for recognition

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

# Upload the voice samples
# These represent the "training" data I used for Francis and Xavier
francis_sample = genai.upload_file("G:\My Drive\pers\Python\training_data\Francis\Francis_straight.wav")
xavier_sample = genai.upload_file("G:\My Drive\pers\Python\training_data\Xavier\Xavier_20260113_013856.wav")
mystery_sample = genai.upload_file("G:\My Drive\pers\Python\training_data\Xavier (1).wav")

# Initialize the model (e.g., Gemini 1.5 Flash or Gemini 3 Flash)
model = genai.GenerativeModel("gemini-1.5-flash")

# The Prompt: This tells the model to perform Speaker Identification
prompt = """
I have provided two reference audio files:
1. Francis's voice (Francis_straight.wav)
2. Xavier's voice (Xavier_20260113_013856.wav)

Please analyze the third audio file (Xavier (1).wav) and tell me which of the two 
original speakers it matches based on their unique vocal characteristics.
"""

response = model.generate_content([prompt, francis_sample, xavier_sample, mystery_sample])

print(response.text)