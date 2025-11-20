
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const classifyBtn = document.getElementById('classifyBtn');
const loader = document.getElementById('loader');
const resultContainer = document.getElementById('resultContainer');
const resultLabel = document.getElementById('resultLabel');
const resultConfidence = document.getElementById('resultConfidence');
const errorMessage = document.getElementById('errorMessage');

let selectedFile = null;
uploadArea.addEventListener('click', function() {
    fileInput.click();
});

// Handle file selection
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = function(event) {
            previewImage.src = event.target.result;
            previewContainer.style.display = 'block';
            classifyBtn.disabled = false;
            resultContainer.style.display = 'none';
            errorMessage.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
});

// Handle classification
classifyBtn.addEventListener('click', async function() {
    if (!selectedFile) return;

    // Show loader
    loader.style.display = 'block';
    resultContainer.style.display = 'none';
    errorMessage.style.display = 'none';
    classifyBtn.disabled = true;

    // Create FormData
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            // Display results
            resultLabel.textContent = `Prediction: ${data.prediction}`;
            resultConfidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            resultContainer.style.display = 'block';
        } else {
            errorMessage.textContent = data.detail || 'An error occurred';
            errorMessage.style.display = 'block';
        }
    } catch (error) {
        errorMessage.textContent = 'Failed to connect to server';
        errorMessage.style.display = 'block';
    } finally {
        loader.style.display = 'none';
        classifyBtn.disabled = false;
    }
});

