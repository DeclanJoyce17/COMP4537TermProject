const express = require('express');
// const { pipeline } = require('@xenova/transformers');
const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs');
const wavDecoder = require('wav-decoder');
const os = require('os');
const path = require('path');
const multer = require('multer');
const router = express.Router();
const { User, ResetToken, APICount } = require("../models");
const { quantize } = require('bitsandbytes');

const MODEL_PATH = path.normalize(path.join(process.cwd(), 'models', 'whisper-base'));
process.env.TRANSFORMERS_CACHE = MODEL_PATH;
process.env.HF_HUB_OFFLINE = "1";
process.env.LOCAL_MODELS_DIR = MODEL_PATH;

// Set up multer for file upload
const upload = multer({ dest: 'uploads/' });

const { exec } = require('child_process');
router.get('/test-ffmpeg', (req, res) => {
    exec('ffmpeg -version', (err, stdout, stderr) => {
        if (err) {
            console.error('FFmpeg test failed:', err);
            return res.status(500).send('FFmpeg not installed');
        }
        // Corrected the response syntax
        res.send(`<pre>FFmpeg installed:\n${stdout}</pre>`);
    });
});

// Route to handle audio file upload and transcription
router.post('/api/transcribe', upload.single('audio'), async (req, res) => {

    console.log("Screw this")

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

let transcriber = null;  // Declare transcriber globally

async function loadModel() {
    if (!transcriber) {
        const { pipeline, AutoProcessor, WhisperForConditionalGeneration } = await import('@xenova/transformers');

        console.log('Loading Whisper model from:', MODEL_PATH);

        try {
            // Verify files exist first
            verifyModelFiles();

            // Manually create processor
            const processor = await AutoProcessor.from_pretrained(MODEL_PATH, {
                local_files_only: true,
                feature_extractor_type: "WhisperFeatureExtractor"
            });

            // Create pipeline with explicit local files
            transcriber = await pipeline('automatic-speech-recognition', {
                model: MODEL_PATH,
                processor: processor,
                local_files_only: true
            });

            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Model loading error:', error);
            throw error;
        }
    }
    return transcriber;
}


// Main transcription function
async function transcribeAudio(audioPath) {
    process.env.HF_HUB_OFFLINE = "1"; // Force offline mode
    process.env.HF_HUB_DISABLE_SYMLINKS = "1";

    const transcriber = await loadModel();
    const tempDir = path.join(os.tmpdir(), `temp-audio-${Date.now()}`);
    fs.mkdirSync(tempDir);

    try {
        // Split into smaller chunks (10 seconds each)
        await splitAudio(audioPath, tempDir, 10);
        const chunkFiles = fs.readdirSync(tempDir)
            .map(file => path.join(tempDir, file))
            .sort();

        const allTranscriptions = [];

        for (const chunkFile of chunkFiles) {
            try {
                const chunkData = fs.readFileSync(chunkFile);
                const decodedAudio = await wavDecoder.decode(chunkData);

                const result = await transcriber(decodedAudio.channelData[0], {
                    language: "en", // Force output in English
                    return_timestamps: false
                });


                allTranscriptions.push(result.text);
            } finally {
                fs.unlinkSync(chunkFile);
                if (global.gc) global.gc();  // Manually trigger garbage collection
            }
        }

        return allTranscriptions.join(' ');
    } catch (error) {
        console.error('Transcription failed:', error);
        throw error;
    } finally {
        // Cleanup
        try {
            fs.unlinkSync(audioPath);
            fs.rmdirSync(tempDir);
        } catch (cleanupError) {
            console.error('Cleanup failed:', cleanupError);
        }
    }
}

async function splitAudio(audioPath, outputDir, chunkSeconds = 10) {
    return new Promise((resolve, reject) => {
        ffmpeg(audioPath)
            .audioChannels(1)
            .audioFrequency(16000)
            .outputOptions([
                '-f', 'segment',
                '-segment_time', chunkSeconds.toString(),
                '-c:a', 'pcm_s16le',  // 16-bit WAV format
                '-ar', '16000',
                '-ac', '1'
            ])
            .output(path.join(outputDir, 'chunk-%03d.wav'))
            .on('end', resolve)
            .on('error', reject)
            .run();
    });
}

function verifyModelFiles() {
    const requiredFiles = [
        'preprocessor_config.json',
        'tokenizer_config.json',
        'config.json',
        path.join('onnx', 'encoder_model.onnx')
    ];

    requiredFiles.forEach(file => {
        const fullPath = path.join(MODEL_PATH, file);
        if (!fs.existsSync(fullPath)) {
            throw new Error(`Missing required model file: ${fullPath}`);
        }
    });
}

// Call this before loadModel()
verifyModelFiles();

module.exports = router;