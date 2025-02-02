import torch
import whisperx
import numpy as np
import wave
import io
import gc
import tempfile
from flask import Flask, render_template
from flask_socketio import SocketIO
import json
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load WhisperX Model (Using base for faster processing)
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model = whisperx.load_model("base", device, compute_type=compute_type)
print(f"Loaded WhisperX model on {device} with {compute_type} compute type.")

# Buffer to store audio chunks
audio_buffer = []
last_audio_time = time.time()

@socketio.on("audio_stream")
def handle_audio(audio_data):
    """Receive and buffer real-time audio from WebRTC before transcribing."""
    global audio_buffer, last_audio_time

    try:
        print(f"Received audio buffer length: {len(audio_data)} bytes")

        # Convert raw PCM to Int16 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Append chunk to buffer
        audio_buffer.append(audio_np)
        last_audio_time = time.time()  # Reset last audio time

    except Exception as e:
        print(f"❌ Error buffering audio: {str(e)}")

@socketio.on("process_audio")
def process_audio():
    """Process buffered audio when speech is finished."""
    global audio_buffer

    if not audio_buffer:
        print("⚠️ No audio in buffer.")
        socketio.emit("transcription", {"text": "(No speech detected)"})
        return

    try:
        # Merge all buffered chunks into one array
        merged_audio = np.concatenate(audio_buffer)
        audio_buffer = []  # Reset buffer

        # Save merged audio as WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(16000)  # 16kHz
            wf.writeframes(merged_audio.tobytes())

        wav_buffer.seek(0)

        # Save WAV file for debugging
        with open("debug_merged_audio.wav", "wb") as f:
            f.write(wav_buffer.getvalue())
        print("✅ Saved merged audio as debug_merged_audio.wav")

        # 🔹 Load and Transcribe with WhisperX
        batch_size = 16
        audio = whisperx.load_audio("debug_merged_audio.wav")
        result = model.transcribe(audio, batch_size=batch_size, language="en")

        # 🔹 Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        del model_a  # Free memory
        gc.collect()

        # 🔹 Save simplified JSON output (without diarization)
        base_name = "debug_merged_audio"
        output_file = f"{base_name}_transcription.json"
        simplified_result = {
            "segments": [
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                }
                for segment in result["segments"]
            ]
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(simplified_result, f, ensure_ascii=False, indent=2)
        print(f"✅ Transcription saved to {output_file}")

        # Send transcription back to frontend
        socketio.emit("transcription", {"text": result["text"], "segments": result["segments"]})

        gc.collect()

    except Exception as e:
        print(f"❌ Error processing audio: {str(e)}")
        socketio.emit("transcription", {"text": f"Error: {str(e)}"})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=9080)
