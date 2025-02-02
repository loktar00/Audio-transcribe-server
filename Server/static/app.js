const socket = io.connect("http://192.168.1.3:9080");

let audioContext;
let scriptProcessor;
let mediaStream;
let source;
let silenceTimer = null; // Detect when user stops speaking

document.getElementById("start-btn").addEventListener("click", async () => {
    try {
        // ✅ Get user microphone input
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,   // Mono
                sampleRate: 16000, // 16kHz sample rate
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        });

        audioContext = new AudioContext({ sampleRate: 16000 });
        source = audioContext.createMediaStreamSource(mediaStream);

        // ✅ Create ScriptProcessor to capture raw PCM data
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        scriptProcessor.onaudioprocess = (event) => {
            let inputBuffer = event.inputBuffer.getChannelData(0);

            // Convert Float32Array to Int16 PCM (Little Endian)
            let pcmData = new DataView(new ArrayBuffer(inputBuffer.length * 2));
            for (let i = 0; i < inputBuffer.length; i++) {
                let sample = Math.max(-1, Math.min(1, inputBuffer[i])); // Clamping
                pcmData.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            }

            // ✅ Send PCM data to server via WebSocket
            socket.emit("audio_stream", pcmData.buffer);

            // ✅ Reset Silence Timer (Wait for user to stop speaking)
            clearTimeout(silenceTimer);
            silenceTimer = setTimeout(() => {
                console.log("⏳ Processing buffered speech...");
                socket.emit("process_audio"); // Tell server to transcribe
            }, 2000); // 2 seconds after last speech
        };

        document.getElementById("start-btn").disabled = true;
        document.getElementById("stop-btn").disabled = false;

    } catch (err) {
        alert("Microphone access denied: " + err.message);
        console.error("Error accessing microphone:", err);
    }
});

document.getElementById("stop-btn").addEventListener("click", () => {
    scriptProcessor.disconnect();
    source.disconnect();
    mediaStream.getTracks().forEach(track => track.stop()); // Stop mic
    document.getElementById("start-btn").disabled = false;
    document.getElementById("stop-btn").disabled = true;
    socket.emit("process_audio"); // Process any remaining audio
});

// Receive transcription from the server
socket.on("transcription", (data) => {
    console.log("Transcription:", data.text);
    document.getElementById("output").innerText = data.text;
});
