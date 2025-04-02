const express = require('express');
const { pipeline } = require('@huggingface/transformers');
const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs');
const wavDecoder = require('wav-decoder');
const os = require('os');
const path = require('path');
const multer = require('multer');
const router = express.Router();
const { User, ResetToken, APICount } = require("../models");


// Constants
const CHUNK_DURATION = 30; // Split audio into 30-second chunks
const MODEL_PATH = path.join(process.cwd(), '/models/whisper-small'); // Ensure correct model path

// Set up multer for file upload
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
        res.json({ text: transcription }); // Send transcription result back`
    } catch (error) {
        console.error('Error during transcription:', error);
        res.status(500).json({ error: 'Error during transcription' }); // Handle errors
    }
});

// Load model once and reuse it
let transcriber;
async function loadModel() {
    if (!transcriber) {
        transcriber = await pipeline('automatic-speech-recognition', MODEL_PATH);
    }
    return transcriber;
}

// Main transcription function
async function transcribeAudio(audioPath) {
    const transcriber = await loadModel();
    const tempDir = path.join(os.tmpdir(), `temp-audio-${Date.now()}`);
    fs.mkdirSync(tempDir);

    try {
        await splitAudio(audioPath, tempDir);
        const chunkFiles = fs.readdirSync(tempDir).map(file => path.join(tempDir, file));

        const allTranscriptions = [];
        for (const chunkFile of chunkFiles) {
            allTranscriptions.push(await transcribeChunk(transcriber, chunkFile));
            fs.unlinkSync(chunkFile); // Delete chunk after transcription
        }

        fs.unlinkSync(audioPath); // Delete the uploaded file after processing
        fs.rmdirSync(tempDir); // Remove the temp directory

        return allTranscriptions.join(' '); // Combine all transcriptions
    } catch (error) {
        console.error('Error during transcription:', error);
        return 'Error processing audio file.';
    }
}

// Function to split audio into chunks
async function splitAudio(audioPath, tempDir) {
    return new Promise((resolve, reject) => {
        ffmpeg(audioPath)
            .audioChannels(1)
            .audioFrequency(16000)
            .outputOptions(['-f', 'segment', '-segment_time', CHUNK_DURATION.toString(), '-reset_timestamps', '1'])
            .output(path.join(tempDir, 'chunk-%03d.wav'))
            .on('end', resolve)
            .on('error', reject)
            .run();
    });
}

// Function to transcribe each chunk of audio
async function transcribeChunk(transcriber, chunkFile) {
    const chunkData = fs.readFileSync(chunkFile);
    const decodedAudio = await wavDecoder.decode(chunkData);

    const result = await transcriber(decodedAudio.channelData[0], {
        task: 'translate',
        language: 'en',
        return_timestamps: true,
    });

    return result.chunks ? result.chunks.map(chunk => chunk.text).join(' ') : result.text;
}

module.exports = router;