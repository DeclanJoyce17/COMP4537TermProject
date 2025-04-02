// Recording variables
const site = "https://comp4537termproject-2.onrender.com";
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
        audioChunks = [];

        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
                sampleRate: 44100,
                channelCount: 1
            }
        });

        mediaRecorder = new MediaRecorder(audioStream);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            try {
                recordingStatus.textContent = 'Processing recording...';

                if (audioChunks.length === 0) {
                    throw new Error('No audio data recorded');
                }

                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const file = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
                await prepareAndSendAudio(file);
            } catch (error) {
                console.error('Processing error:', error);
                recordingStatus.textContent = 'Error processing recording';
            } finally {
                cleanupRecording();
            }
        };

        mediaRecorder.start();
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

// Stop recording
async function stopRecording() {
    if (!isRecording) return;
    try {
        mediaRecorder.requestData();
        mediaRecorder.stop();
        isRecording = false;
        recordButton.textContent = 'Start Recording';
        recordButton.classList.remove('recording');
    } catch (error) {
        console.error('Stop recording error:', error);
        recordingStatus.textContent = 'Error stopping recording';
        cleanupRecording();
    }
}

// Upload an audio file
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
        await prepareAndSendAudio(file);
    } catch (error) {
        console.error('Upload error:', error);
        transcriptionOutput.textContent = 'Error: Failed to transcribe audio';
    } finally {
        uploadButton.disabled = false;
        uploadButton.textContent = 'Transcribe File';
    }
}

// Process and send audio to server
async function prepareAndSendAudio(file) {
    const formData = new FormData();
    formData.append('audio', file);

    // try {
    //     console.log("Testing FFmpeg installation...");
    //     const ffmpegTest = await fetch(`/transcribe/test-ffmpeg`, {
    //         method: 'GET',
    //         credentials: 'include',
    //         headers: { 'Accept': 'application/json' }
    //     });

    //     if (!ffmpegTest.ok) {
    //         throw new Error("FFmpeg test failed");
    //     }

    //     const ffmpegResult = await ffmpegTest.text();
    //     console.log("FFmpeg test successful:", ffmpegResult);

    // } catch (error) {
    //     console.error("FFmpeg test failed:", error);
    //     transcriptionOutput.textContent = "Error: Server processing unavailable (FFmpeg error)";
    //     return;
    // }

    console.log("Starting fetch request...");

    try {
        const response = await fetch(`/transcribe/api/transcribe`, {
            method: 'POST',
            body: formData,
            keepalive: true,
            credentials: 'include'
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

// Clean up resources
function cleanupRecording() {
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
}
