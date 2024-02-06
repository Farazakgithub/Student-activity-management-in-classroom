function predict() {
    var fileInput = document.getElementById('file');
    var formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('predictionResult').innerHTML = 'Predicted Activity: ' + data.prediction;
    })
    .catch(error => console.error('Error:', error));
}
