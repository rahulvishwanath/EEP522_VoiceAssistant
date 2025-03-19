import sounddevice as sd
import queue
import numpy as np
from faster_whisper import WhisperModel

# Initialize queue for real-time audio
q = queue.Queue()

# Load Whisper model (optimized for Raspberry Pi)
model = WhisperModel("rhasspy/faster-whisper-tiny-int8", compute_type="int8")

def transcribe_audio_stream(audio_data):
    """ Convert audio bytes to NumPy array & transcribe with Whisper """
    # Convert raw byte data to NumPy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize audio

    # Transcribe with Whisper
    segments, _ = model.transcribe(audio_np, language="en")

    # Extract & print recognized text
    for segment in segments:
        print(f"🗣 Recognized: {segment.text}")
        return segment.text  # Return first recognized segment

def callback(indata, frames, time, status):
    """ Capture audio and put into queue """
    if status:
        print(f"Audio Input Error: {status}")
    q.put(indata.copy())  # Store audio data in queue

# Start audio stream for real-time transcription
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                        channels=1, callback=callback):
    print("🎤 Speak now...")
    while True:
        try:
            # Get audio data from queue
            data = q.get()
            
            # Transcribe the audio
            result = transcribe_audio_stream(data)

            # Print the result
            if result:
                print(f"✅ Recognized: {result}")

        except KeyboardInterrupt:
            print("🛑 Stopping transcription...")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
