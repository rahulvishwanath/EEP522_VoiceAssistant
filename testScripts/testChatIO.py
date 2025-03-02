import sounddevice as sd
import queue
import vosk
import json
import os
import subprocess

# Paths for Vosk and Piper
VOSK_MODEL_PATH = "../models/vosk-model-small-en-us-0.15"
PIPER_MODEL_PATH = "../models/piper/en_US-amy-low.onnx"
PIPER_EXE_PATH = os.path.join(os.path.dirname(__file__), '../models/piper/piper')

# Initialize Vosk STT
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

model = vosk.Model(VOSK_MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, 16000)

# Function to generate and stream speech output using Piper
def text_to_speech_streaming(text):
    print("Exe Path ", PIPER_EXE_PATH)
    command = f'echo "{text}" | {PIPER_EXE_PATH} --model {PIPER_MODEL_PATH} --output-raw | aplay -r 16000 -f S16_LE -t raw -'
    subprocess.run(command, shell=True)

# Chatbot logic (replace this with AI model logic)
def chatbot_response(text):
    return f"You said: {text}. How can I assist you?"

# Main loop: Real-time speech recognition and response
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
    print("? Speak now...")
    while True:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            user_text = result['text']
            print(f"? Recognized: {user_text}")

            if user_text:
                # Generate chatbot response
                response_text = chatbot_response(user_text)
                print(f"? Chatbot Response: {response_text}")

                # Speak the response in real-time using Piper
                text_to_speech_streaming(response_text)
