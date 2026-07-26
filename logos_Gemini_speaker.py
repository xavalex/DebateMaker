import google.generativeai as genai
import os

# Configure your API key
genai.configure(api_key="AIzaSyB3oXNknwh-gLzscgykfTXIYoGp1Jb7DWs")

# Configuration
DATA_DIR = r'G:\My Drive\pers\Python\training_data'
DURATION = 5  # Seconds to record for recognition

# Upload the voice samples
# These represent the "training" data I used for Francis and Xavier
francis_sample = genai.upload_file(r"G:\My Drive\pers\Python\training_data\Francis\Francis_straight.wav")
xavier_sample = genai.upload_file(r"G:\My Drive\pers\Python\training_data\Xavier\Xavier_20260113_013856.wav")
mystery_sample = genai.upload_file(r"G:\My Drive\pers\Python\training_data\Feancis_15sec.ogg")

# Initialize the model (e.g., Gemini 1.5 Flash or Gemini 3 Flash)
model = genai.GenerativeModel("gemini-2.5-flash")

# The Prompt: This tells the model to perform Speaker Identification
prompt = """
I have provided two reference audio files:
1. Francis's voice (Francis_straight.wav)
2. Xavier's voice (Xavier_20260113_013856.wav)

Please analyze the third audio file (Feancis_15sec.ogg) and tell me which of the two 
original speakers it matches based on their unique vocal characteristics.
"""

response = model.generate_content([prompt, francis_sample, xavier_sample, mystery_sample])

print(response.text)