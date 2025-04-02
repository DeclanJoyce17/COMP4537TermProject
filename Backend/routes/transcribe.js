const express = require('express');
const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs');
const wavDecoder = require('wav-decoder');
const os = require('os');
const path = require('path');
const multer = require('multer');
const router = express.Router();
const { User, ResetToken, APICount } = require("../models");
const upload = multer({ dest: 'uploads/' });

// Route to handle audio file upload and transcription
router.post('/api/transcribe', upload.single('audio'), async (req, res) => {

    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' }); // Return error if no file was uploaded
    }

    const count = await APICount.findOne({ api: "/transcribe/api/transcribe" });

    try {
        const transcription = await transcribeAudio(req.file.path); // Transcribe the uploaded file
        console.log(transcription)
        res.json({ text: transcription }); // Send transcription result back
    } catch (error) {
        console.error('Error during transcription:', error);
        res.status(500).json({ error: 'Error during transcription' }); // Handle errors
    }

});

// Load model once and reuse it
let transcriber = null;

async function loadModel() {
    if (!transcriber) {
        const transformers = await import('@xenova/transformers');
        const pipeline = transformers.pipeline;

        console.log('Loading Whisper-Medium small from Hugging Face...');

        transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small', {
            quantized: true,
            device: 'cpu',
        });

        console.log('Whisper-Medium small loaded successfully');
    }
    return transcriber;
}

async function transcribeAudio(audioPath) {
    const transcriber = await loadModel();
    let wavPath = path.join(path.dirname(audioPath), 'converted.wav'); // Define wavPath before try block

    try {
        console.log("Processing audio:", audioPath);

        // Convert to WAV format with 16kHz, mono
        await convertToWav(audioPath, wavPath);

        // Read the converted WAV file
        const audioBuffer = fs.readFileSync(wavPath);

        // Decode WAV to PCM float array
        const decodedAudio = await wavDecoder.decode(audioBuffer);
        const float32Array = new Float32Array(decodedAudio.channelData[0]); // Extract first channel

        const result = await transcriber(float32Array, {
            language: "en",
            return_timestamps: false
        });

        console.log("Transcription result:", result.text);
        return result.text;
    } catch (error) {
        console.error("Transcription failed:", error);
        throw error;
    } finally {
        try {
            if (fs.existsSync(audioPath)) fs.unlinkSync(audioPath); // Delete original file
            if (fs.existsSync(wavPath)) fs.unlinkSync(wavPath); // Delete converted WAV file
        } catch (cleanupError) {
            console.error("Failed to delete files:", cleanupError);
        }
    }
}

// Function to convert any audio file to WAV
async function convertToWav(inputPath, outputPath) {
    return new Promise((resolve, reject) => {
        ffmpeg(inputPath)
            .output(outputPath)
            .audioChannels(1) // Mono audio
            .audioFrequency(16000) // 16kHz sample rate
            .format('wav') // Ensure WAV format
            .on('end', resolve)
            .on('error', reject)
            .run();
    });
}

module.exports = router;