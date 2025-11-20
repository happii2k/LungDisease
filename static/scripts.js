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

// Make upload area clickable
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

        if (response.ok && data.success) {
            // Get the prediction label (correct field name)
            const prediction = data.prediction_label || 'Unknown';
            const confidence = data.confidence || 0;
            
            // Determine if it's pneumonia or normal
            const isPneumonia = prediction.toLowerCase().includes('pneumonia');
            
            // Display results with color coding
            resultLabel.innerHTML = `
                <div class="prediction-result ${isPneumonia ? 'pneumonia' : 'normal'}">
                    <div class="prediction-icon">${isPneumonia ? '⚠️' : '✅'}</div>
                    <div class="prediction-text">
                        <h3>${prediction}</h3>
                        <p class="confidence-text">Confidence: ${(confidence * 100).toFixed(2)}%</p>
                    </div>
                </div>
            `;
            
            // Show all probabilities if available
            if (data.all_probabilities) {
                let probText = '<div class="all-probs"><h4>All Predictions:</h4><ul>';
                for (const [label, prob] of Object.entries(data.all_probabilities)) {
                    probText += `<li><strong>${label}:</strong> ${(prob * 100).toFixed(2)}%</li>`;
                }
                probText += '</ul></div>';
                resultConfidence.innerHTML = probText;
            } else {
                resultConfidence.innerHTML = '';
            }
            
            resultContainer.style.display = 'block';
        } else {
            errorMessage.textContent = data.detail || 'Prediction failed';
            errorMessage.style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
        errorMessage.textContent = 'Failed to connect to server. Please try again.';
        errorMessage.style.display = 'block';
    } finally {
        loader.style.display = 'none';
        classifyBtn.disabled = false;
    }
});
