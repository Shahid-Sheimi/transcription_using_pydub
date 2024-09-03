from flask import Flask, request, jsonify, render_template
import requests
import pyaudio
import io
import threading
from pydub import AudioSegment
from pydub.silence import split_on_silence

app = Flask(__name__)

# Configuration
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'  # Replace with your OpenAI API key
WHISPER_API_URL = 'https://api.openai.com/v1/audio/transcriptions'

# Audio stream configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def get_audio_stream():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    return audio, stream

def process_audio_data(audio_data):
    # Convert raw audio data to AudioSegment
    audio_segment = AudioSegment(
        data=audio_data,
        sample_width=2,  # 2 bytes for 16-bit audio
        frame_rate=RATE,
        channels=CHANNELS
    )
    
    # Optionally, split on silence or process further
    # segments = split_on_silence(audio_segment, silence_thresh=-40)
    
    return audio_segment

def transcribe_audio(audio_segment):
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)
    
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'multipart/form-data'
    }
    files = {
        'file': ('audio.wav', buffer, 'audio/wav')
    }
    data = {
        'model': 'whisper-1',
        'response_format': 'text'
    }
    response = requests.post(WHISPER_API_URL, headers=headers, files=files, data=data)
    response.raise_for_status()
    return response.json().get('text', '')

def record_and_transcribe():
    audio, stream = get_audio_stream()
    frames = []
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            if len(frames) * CHUNK > RATE:  # Process every second
                audio_data = b''.join(frames)
                audio_segment = process_audio_data(audio_data)
                transcription = transcribe_audio(audio_segment)
                if transcription:
                    print("Transcription:", transcription)
                frames = []  # Clear frames after sending to API
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_transcription', methods=['POST'])
def start_transcription():
    threading.Thread(target=record_and_transcribe).start()
    return jsonify({'status': 'Recording started'})

if __name__ == '__main__':
    app.run(debug=True)
