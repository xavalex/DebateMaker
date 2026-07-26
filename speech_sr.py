import speech_recognition as sr

# Initialize the recognizer
r = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    print("Say something...")
    # Adjust for ambient noise to improve accuracy
    r.adjust_for_ambient_noise(source, duration=1)
    # Listen for the user's input
    audio_data = r.listen(source)

    try:
        # Use the offline CMU Sphinx recognizer
        text = r.recognize_sphinx(audio_data)
        print(f"You said: {text}")
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print(f"Sphinx error; {e}")