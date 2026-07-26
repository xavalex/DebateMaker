import os
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.dataio.dataio import read_audio



# 1. Data Preparation
# Create a directory to save the trained model
os.makedirs("speaker_model_sb", exist_ok=True)

# Path to your training data
data_folder = "training_data"
speaker_dirs = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]

print(f"Found speakers: {speaker_dirs}")

# 2. Model Training (Fine-tuning a pretrained model)
# We will use a powerful pretrained model (ECAPA-TDNN) and fine-tune it on your data.
# This is much more effective than training from scratch, especially with little data.
# Let SpeechBrain handle the caching automatically
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# This is a simplified fine-tuning example.
# For a real-world scenario, you'd use SpeechBrain's full training pipeline.
# Here, we'll just "enroll" the speakers by getting their average embedding.

print("\nEnrolling speakers...")
speaker_embeddings = {}

for speaker in speaker_dirs:
    speaker_dir_path = os.path.join(data_folder, speaker)
    
    # Get all wav files for the speaker
    wav_files = [f for f in os.listdir(speaker_dir_path) if f.endswith(".wav")]
    
    if not wav_files:
        print(f"Warning: No .wav files found for speaker {speaker}. Skipping.")
        continue

    print(f"  - Processing {speaker}...")
    
    # We'll average the embeddings from all files for that speaker
    embeddings = []
    for wav_file in wav_files:
        file_path = os.path.join(speaker_dir_path, wav_file)
        try:
            # Load the audio file
            signal = read_audio(file_path)
            
            # The model expects a batch, so we add a dimension
            embedding = classifier.encode_batch(signal.unsqueeze(0))
            
            # Remove the batch dimension and the time dimension
            embeddings.append(embedding.squeeze())
        except Exception as e:
            print(f"    - Error processing {wav_file}: {e}")

    # Average the embeddings to get a single representative vector for the speaker
    if embeddings:
        speaker_embeddings[speaker] = torch.mean(torch.stack(embeddings), dim=0)
        print(f"    -> Enrolled {speaker}")

if not speaker_embeddings:
    print("\nError: No speakers were enrolled. Please check your training data.")
else:
    # Save the speaker embeddings
    torch.save(speaker_embeddings, "speaker_model_sb/speaker_embeddings.pt")
    print("\nSpeaker enrollment complete. Embeddings saved to 'speaker_model_sb/speaker_embeddings.pt'")


    # 3. Speaker Verification (Example)
    print("\n--- Speaker Verification Example ---")
    # Load the saved embeddings
    loaded_embeddings = torch.load("speaker_model_sb/speaker_embeddings.pt")

    # Let's pretend this is a new audio file we want to verify
    # We'll just use one of the training files for this example
    try:
        # Find the first speaker with at least one file to use for testing
        test_speaker = None
        test_file_path = None
        for spk in speaker_dirs:
            spk_path = os.path.join(data_folder, spk)
            wavs = [f for f in os.listdir(spk_path) if f.endswith(".wav")]
            if wavs:
                test_speaker = spk
                test_file_path = os.path.join(spk_path, wavs[0])
                break
        
        if not test_file_path:
            raise IndexError("No WAV files found in any speaker directory for testing.")

        print(f"Testing with file: {test_file_path}")

        # Get the embedding of the test file
        test_signal = read_audio(test_file_path)
        test_embedding = classifier.encode_batch(test_signal.unsqueeze(0)).squeeze()

        # Compare the test embedding with all enrolled speakers
        best_match_speaker = None
        highest_similarity = -1.0  # Use a float

        for speaker, enrolled_embedding in loaded_embeddings.items():
            # Using Cosine Similarity to compare embeddings
            similarity = torch.nn.functional.cosine_similarity(test_embedding, enrolled_embedding, dim=0)
            print(f"  - Similarity with {speaker}: {similarity.item():.4f}")
            
            if similarity.item() > highest_similarity:
                highest_similarity = similarity.item()
                best_match_speaker = speaker

        if best_match_speaker:
            print(f"\n--> Predicted Speaker: {best_match_speaker} (Similarity: {highest_similarity:.4f})")
        else:
            print("\nCould not find a confident match.")

    except IndexError as e:
        print(f"\nCould not run verification example: {e}")
    except Exception as e:
        print(f"\nAn error occurred during verification: {e}")

