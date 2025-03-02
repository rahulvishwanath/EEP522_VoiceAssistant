import sounddevice as sd
import numpy as np
import wave

fs = 16000  # Sample rate
duration = 5  # seconds
filename = "bluetooth_mic_test.wav"

def record_audio():
    print("? Recording for 5 seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    print("? Recording complete.")

    # Save audio to file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

    print(f"? Saved to {filename}")

def play_audio():
    print("? Playing recorded audio...")
    wf = wave.open(filename, "rb")
    audio_data = wf.readframes(fs * duration)
    sd.play(np.frombuffer(audio_data, dtype=np.int16), samplerate=fs)
    sd.wait()
    print("? Playback complete.")

record_audio()
play_audio()
