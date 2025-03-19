# **Embedded AI Voice Assistant on Raspberry Pi**

## **Overview**
This project implements an **offline AI-powered voice assistant** on a **Raspberry Pi 4** using:
- **Vosk** for **Speech-to-Text (STT)**
- **GPT-3.5** for **intent classification & chatbot responses**
- **Pinecone** for **storing and retrieving personal data**
- **Piper** for **Text-to-Speech (TTS)**
- **Sounddevice** for **real-time audio processing**

Unlike cloud-based solutions, this assistant **processes everything locally** for **privacy**, **low latency**, and **offline functionality**.

## **Features**
**Speech Recognition (STT)** using **Vosk**  
**Intent Classification** with **GPT-3.5** (Future: Local Language model)  
**Personalized Data Handling** using **Pinecone Vector Database**  
**AI Chatbot Responses** using **GPT-3.5 API**  
**Natural-Sounding Voice Output (TTS)** using **Piper**  
**Real-Time Audio Streaming** with **Sounddevice**  

---

## **Project Structure**
```
EEP522_VoiceAssistant/
│── models/                # Stores models for Vosk (STT) and Piper (TTS)
│    ├── vosk-model-small-en-us-0.15/    # Vosk STT model (offline speech recognition)
│    ├── piper/                          # Piper TTS model for speech synthesis
│
│── testScripts/           # Contains individual component test scripts
│    ├── testAudio.py         # Tests audio input/output
│    ├── testChatIO.py        # Tests chatbot integration
│    ├── testChatbot.py       # Standalone chatbot implementation
│    ├── testWhisperInput.py  # Tests Whisper-based STT (alternative)
│    ├── testvosk.py          # Tests Vosk STT
│
│── personalAssistant.py   # Main script for running the AI assistant
│── README.md              # Documentation (this file)
│── LICENSE                # License information
│── .gitignore             # Ignore unnecessary files in GitHub
│── last_vector_id.json     # Stores the last Pinecone vector ID
```

---

## **Installation Guide**

### **1. Install Additional Packages**
Ensure the required libraries are installed:
```bash
pip install sounddevice vosk openai pinecone-client json dateparser subprocess
```

### **2. Download and Set Up Models**
Download the **Vosk Speech-to-Text model** and **Piper Text-to-Speech model** into the **models/** folder.
```bash
mkdir -p models/vosk-model-small-en-us-0.15
mkdir -p models/piper
```
Ensure the correct **Piper** model (`en_US-amy-low.onnx`) is placed in **models/piper/**.

---

## **Running the Assistant**
Once everything is set up, run the **main script** to start the AI assistant:
```bash
python3 personalAssistant.py
```
The assistant will start listening for voice input and respond in **real-time**.

---

## **Testing Individual Components**
You can test each component separately using the scripts in the `testScripts/` folder.

- **Test Audio Input/Output:**
  ```bash
  python3 testAudio.py
  ```
- **Test Chatbot:**
  ```bash
  python3 testChatbot.py
  ```
- **Test STT (Vosk):**
  ```bash
  python3 testvosk.py
  ```
- **Test Whisper STT (Optional):**
  ```bash
  python3 testWhisperInput.py
  ```

---

## **How It Works**
### **1. Speech-to-Text (Vosk)**
- Captures **voice input** using a **USB microphone**.
- Converts speech to text using **Vosk STT**.

### **2. Intent Classification (GPT-3.5)**
- Determines if the query is:
  - **General Query**
  - **Personal Data Store**
  - **Personal Data Retrieve**
  - **Prompt Injection** (filtered out)
  - **Offensive Intent** (filtered out)

### **3. Personal Data Storage & Retrieval (Pinecone)**
- Stores **structured data** like reminders:
  ```
  "Remind me to buy milk tomorrow at 10 AM"
  ```
  Stored as:
  ```json
  {
    "type": "Reminder - Groceries",
    "task": "Buy milk",
    "time": "03/14/25 10:00 AM"
  }
  ```
- Retrieves stored information when queried.

### **4. Chatbot Responses (GPT-3.5)**
- Answers general questions using **OpenAI API**.
- Future enhancement: **Fine-tuned local models** (Llama).

### **5. Text-to-Speech (Piper)**
- Converts text responses to **natural-sounding speech**.
- Streams audio output **in real-time** using:
  ```bash
  echo "Hello, how can I help?" | ./piper --model en_US-amy-low.onnx --output-raw | aplay -r 16000 -f S16_LE -t raw -
  ```

---

## **Future Enhancements**
🚀 **Upgrade STT to Whisper-Tiny Int8** for better speech recognition.  
🚀 **Implement Local Intent Classifier** to replace GPT-3.5 API.  
🚀 **Replace OpenAI Chatbot with Llama Models** for full offline functionality.  
🚀 **Custom Wake Word Detection** ("Hey Pi") for hands-free activation.  

---

## **Troubleshooting**
### **1. Vosk Model Not Loading**
Ensure the **Vosk model** is placed in `models/vosk-model-small-en-us-0.15/`.

### **2. Piper Not Generating Speech**
Check the Piper model path and permissions:
```bash
ls -l models/piper/
```
If missing, **re-download** the correct model.

### **3. Microphone Not Working**
Run:
```bash
arecord -l
```
Ensure the correct **audio input device** is selected.

---

## **License**
This project is released under the **MIT License**.
