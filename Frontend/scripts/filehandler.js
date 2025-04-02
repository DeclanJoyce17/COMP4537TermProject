// Recording variables
const site = "https://comp4537termproject-1-181a.onrender.com"
let mediaRecorder;
let audioStream;
let audioChunks = [];
let isRecording = false;
const recordButton = document.getElementById('recordButton');
const recordingStatus = document.getElementById('recordingStatus');
const uploadButton = document.getElementById('uploadButton');
const audioFileInput = document.getElementById('audioFile');
const transcriptionOutput = document.getElementById('transcription');

// Initialize event listeners
// recordButton.addEventListener('click', toggleRecording);
recordButton.addEventListener('click', (event) => {
    event.preventDefault(); // Stop page refresh
    toggleRecording().catch(error => console.error('Unhandled error:', error));
});

uploadButton.addEventListener('click', (event) => {
    event.preventDefault(); // Stop page refresh
    uploadAudio().catch(error => console.error('Unhandled error:', error));
});


document.addEventListener("beforeunload", (event) => {
    console.log("Something is triggering a page unload!");
    event.preventDefault();
});

document.addEventListener("submit", (event) => {
    event.preventDefault();
    console.log("A form tried to submit, but we blocked it!");
});

// Toggle recording function
async function toggleRecording() {
    if (isRecording) {
        await stopRecording();
    } else {
        await startRecording();
    }
}

// Start recording
async function startRecording() {
    try {
        // Clear previous chunks
        audioChunks = [];

        // Request microphone access with optimal settings
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false, // Disabled for better quality
                noiseSuppression: false, // Disabled for better quality
                autoGainControl: false,  // Disabled for better quality
                sampleRate: 44100,       // CD quality
                channelCount: 1          // Mono is better for speech
            }
        });

        // Initialize recorder with no timeslice to get everything at the end
        mediaRecorder = new MediaRecorder(audioStream);

        // Collect all chunks during recording
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        // Handle when recording stops
        mediaRecorder.onstop = async () => {
            try {
                recordingStatus.textContent = 'Processing recording...';

                // Ensure we have audio chunks
                if (audioChunks.length === 0) {
                    throw new Error('No audio data recorded');
                }

                // Combine all chunks into a single Blob
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

                // Convert to WAV for better compatibility
                const wavBlob = await convertToWav(audioBlob);

                // Send to server
                await sendAudioToServer(wavBlob, 'recording.wav');

            } catch (error) {
                console.error('Processing error:', error);
                recordingStatus.textContent = 'Error processing recording';
            } finally {
                cleanupRecording();
            }
        };

        // Start recording with NO timeslice to get everything at the end
        mediaRecorder.start();

        // Update UI
        isRecording = true;
        recordButton.textContent = 'Stop Recording';
        recordButton.classList.add('recording');
        recordingStatus.textContent = 'Recording - Speak now...';

    } catch (error) {
        console.error('Recording error:', error);
        recordingStatus.textContent = 'Error: Could not access microphone';
        cleanupRecording();
    }
}

// Stop recording - now requests final data explicitly
async function stopRecording() {
    if (!isRecording) return;

    try {
        // Request any remaining data
        mediaRecorder.requestData();

        // Stop the recorder which will trigger onstop
        mediaRecorder.stop();

        // Update UI immediately
        isRecording = false;
        recordButton.textContent = 'Start Recording';
        recordButton.classList.remove('recording');

    } catch (error) {
        console.error('Stop recording error:', error);
        recordingStatus.textContent = 'Error stopping recording';
        cleanupRecording();
    }
}

// [Rest of the functions remain the same...]

// Clean up recording resources
function cleanupRecording() {
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
}

// Audio conversion to WAV
async function convertToWav(audioBlob) {
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        // Convert to mono if needed
        let sourceBuffer;
        if (audioBuffer.numberOfChannels > 1) {
            sourceBuffer = convertToMono(audioBuffer);
        } else {
            sourceBuffer = audioBuffer.getChannelData(0);
        }

        // Create WAV file
        const wavHeader = createWavHeader(sourceBuffer.length, audioBuffer.sampleRate);
        const wavData = new Uint8Array(wavHeader.byteLength + sourceBuffer.length * 2);

        // Add header
        wavData.set(new Uint8Array(wavHeader), 0);

        // Add audio data
        const view = new DataView(wavData.buffer);
        let offset = wavHeader.byteLength;

        // Write 16-bit PCM data
        for (let i = 0; i < sourceBuffer.length; i++) {
            const sample = Math.max(-1, Math.min(1, sourceBuffer[i])); // clamp
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }

        return new Blob([wavData], { type: 'audio/wav' });

    } catch (error) {
        console.error('Conversion error:', error);
        throw new Error('Failed to convert audio to WAV');
    }
}

// Helper functions for WAV conversion...

// Upload audio file (unchanged from previous version)
async function uploadAudio() {

    const file = audioFileInput.files[0];
    if (!file) {
        alert('Please select an audio file first');
        return;
    }

    uploadButton.disabled = true;
    uploadButton.textContent = 'Processing...';
    transcriptionOutput.textContent = 'Transcribing audio...';

    try {
        await sendAudioToServer(file, file.name);
    } catch (error) {
        console.error('Upload error:', error);
        transcriptionOutput.textContent = 'Error: Failed to transcribe audio';
    } finally {
        uploadButton.disabled = false;
        uploadButton.textContent = 'Transcribe File';
    }
}

// Send audio to server (unchanged from previous version)
async function sendAudioToServer(file, filename) {
    const formData = new FormData();
    formData.append('audio', file);

    console.log("Starting fetch request...");

    try {
        const response = await fetch(`${site}/transcribe/api/transcribe`, {
            method: 'POST',
            body: formData,
            keepalive: true
        });

        console.log("Fetch response received:", response);

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Transcription result:", result);
        transcriptionOutput.textContent = result.text;
    } catch (error) {
        console.error("Fetch error:", error);
        transcriptionOutput.textContent = "Error: Failed to transcribe audio";
    }
}


// Helper functions for WAV conversion
function convertToMono(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    const result = new Float32Array(length);

    for (let i = 0; i < length; i++) {
        let sum = 0;
        for (let channel = 0; channel < numChannels; channel++) {
            sum += audioBuffer.getChannelData(channel)[i];
        }
        result[i] = sum / numChannels;
    }

    return result;
}

function createWavHeader(dataLength, sampleRate) {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);

    // RIFF identifier
    writeString(view, 0, 'RIFF');
    // RIFF chunk length
    view.setUint32(4, 36 + dataLength * 2, true);
    // RIFF type
    writeString(view, 8, 'WAVE');
    // Format chunk identifier
    writeString(view, 12, 'fmt ');
    // Format chunk length
    view.setUint32(16, 16, true);
    // Sample format (raw PCM)
    view.setUint16(20, 1, true);
    // Channel count (mono)
    view.setUint16(22, 1, true);
    // Sample rate
    view.setUint32(24, sampleRate, true);
    // Byte rate (sample rate * block align)
    view.setUint32(28, sampleRate * 2, true);
    // Block align (channel count * bytes per sample)
    view.setUint16(32, 2, true);
    // Bits per sample
    view.setUint16(34, 16, true);
    // Data chunk identifier
    writeString(view, 36, 'data');
    // Data chunk length
    view.setUint32(40, dataLength * 2, true);

    return header;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}