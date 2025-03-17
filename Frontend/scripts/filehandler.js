async function uploadAudio() {
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file.');
        return;
    }

    const formData = new FormData();
    formData.append('audio', file);

    try {
        const response = await fetch('/transcribe/api/transcribe', {
            method: 'POST',
            body: formData
        });


        const data = await response.json();
        if (data.text) {
            document.getElementById('transcription').innerText = data.text;
        } else {
            document.getElementById('transcription').innerText = 'Error transcribing audio.';
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('transcription').innerText = 'Error transcribing audio.';
    }
}
