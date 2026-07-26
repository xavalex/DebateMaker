import pyaudio

p = pyaudio.PyAudio()

print("--- PyAudio Device Information ---")
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

if numdevices == 0:
    print("No audio devices found.")
else:
    print(f"Found {numdevices} audio devices:")
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        print(f"  Device {i}:")
        print(f"    Name: {device_info.get('name')}")
        print(f"    Max Input Channels: {device_info.get('maxInputChannels')}")
        print(f"    Max Output Channels: {device_info.get('maxOutputChannels')}")
        print(f"    Default Sample Rate: {device_info.get('defaultSampleRate')}")

print("\n--- Default Input Device ---")
try:
    default_input = p.get_default_input_device_info()
    print(f"  Name: {default_input.get('name')}")
    print(f"  Index: {default_input.get('index')}")
    print(f"  Max Input Channels: {default_input.get('maxInputChannels')}")
except IOError:
    print("No default input device found.")

p.terminate()
print("\n--- End of Report ---")
