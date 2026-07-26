
from google.cloud import speech

print("just before def")
def transcribe_diarization_gcs_beta(audio_uri: str) -> bool:
    """Transcribe a remote audio file (stored in Google Cloud Storage) using speaker diarization.
    Args:
        audio_uri (str): The Google Cloud Storage path to an audio file.
            E.g., gs://[BUCKET]/[FILE]
    Returns:
        True if the operation successfully completed, False otherwise.
    """

    client = speech.SpeechClient()
    # Enhance diarization config with more speaker counts and details
    speaker_diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,  # Set minimum number of speakers
        max_speaker_count=2,  # Adjust max speakers based on expected number of speakers
    )

    # Configure recognition with enhanced audio settings
    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        sample_rate_hertz=8000,
        diarization_config=speaker_diarization_config,
    )
    print("just before diarisation")
    # Set the remote path for the audio file
    audio = speech.RecognitionAudio(
        uri=audio_uri,)

    # Use non-blocking call for getting file transcription
    response = client.long_running_recognize(
        config=recognition_config, audio=audio
    ).result(timeout=300)

    # The transcript within each result is separate and sequential per result.
    # However, the words list within an alternative includes all the words
    # from all the results thus far. Thus, to get all the words with speaker
    # tags, you only have to take the words list from the last result
    result = response.results[-1]
    words_info = result.alternatives[0].words

    # Print the output
    for word_info in words_info:
        print(f"word: '{word_info.word}', speaker_tag: {word_info.speaker_tag}")
    return True
transcribe_diarization_gcs_beta("vader_faith.wav")
