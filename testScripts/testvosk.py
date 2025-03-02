import sounddevice as sd
import queue
import vosk
import json

model_path = "../models/vosk-model-small-en-us-0.15"  # Download the model beforehand
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

model = vosk.Model(model_path)
recognizer = vosk.KaldiRecognizer(model, 16000)

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
    print("? Speak now...")
    while True:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            print(f"? Recognized: {result['text']}")
