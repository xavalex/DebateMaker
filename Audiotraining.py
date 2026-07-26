import sys
import os
import matplotlib

from pyAudioAnalysis import audioAnalysis as aA
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import audioTrainTest as aT


# The trainClassifierWrapper expects a list of directories, one for each class (speaker)
# I will assume you have subdirectories in 'training_data' like 'speaker1', 'speaker2', etc.
# Please adjust this list to match your directory structure.
training_dirs = [os.path.join("training_data", d) for d in os.listdir("training_data") if os.path.isdir(os.path.join("training_data", d))]

if not training_dirs:
    print("Error: No training data subdirectories found in 'training_data/'.")
    print("Please create subdirectories for each speaker (e.g., 'training_data/speaker1/', 'training_data/speaker2/').")
else:
    aA.trainClassifierWrapper(directories=training_dirs, method="svm", model_name="speaker_model", beat_feats=False)
    print("Model training complete. Model saved as 'speaker_model.svm'.")

# Path to the new audio file you want to classify
# new_audio_file = "new_audio.wav"

# Path to the trained SVM model
# trained_model = "speaker_model.svm"

# Use the classifyFile function
# This will output a probability distribution over the trained speakers
# and identify the most likely speaker.
# aA.classifyFileWrapper(new_audio_file, "svm", trained_model)