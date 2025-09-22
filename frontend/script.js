class DigitCanvas {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        
        this.setupCanvas();
        this.setupEventListeners();
    }
    
    setupCanvas() {
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.lineWidth = 8;
        this.ctx.strokeStyle = '#000';
        this.ctx.fillStyle = '#fff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    setupEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });
        
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });
        
        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {});
            this.canvas.dispatchEvent(mouseEvent);
        });
    }
    
    startDrawing(e) {
        this.isDrawing = true;
        const rect = this.canvas.getBoundingClientRect();
        this.lastX = e.clientX - rect.left;
        this.lastY = e.clientY - rect.top;
    }
    
    draw(e) {
        if (!this.isDrawing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(currentX, currentY);
        this.ctx.stroke();
        
        this.lastX = currentX;
        this.lastY = currentY;
    }
    
    stopDrawing() {
        this.isDrawing = false;
    }
    
    clear() {
        this.ctx.fillStyle = '#fff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    getImageData() {
        // Resize canvas to 28x28 for MNIST format
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        
        tempCtx.drawImage(this.canvas, 0, 0, 28, 28);
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        
        // Convert to grayscale and flatten
        const pixels = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
            const r = imageData.data[i];
            const g = imageData.data[i + 1];
            const b = imageData.data[i + 2];
            const gray = (r + g + b) / 3;
            pixels.push(gray);
        }
        
        return pixels;
    }
}

class PredictionAPI {
    constructor() {
        this.baseUrl = window.location.origin;
    }
    
    async predict(imageData) {
        try {
            const response = await fetch(`${this.baseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }
    
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check error:', error);
            return { status: 'unhealthy', model_loaded: false };
        }
    }
}

class UI {
    constructor() {
        this.predictionResult = document.getElementById('predictionResult');
        this.confidenceBars = document.getElementById('confidenceBars');
        this.predictBtn = document.getElementById('predictBtn');
        this.clearBtn = document.getElementById('clearBtn');
    }
    
    showLoading() {
        this.predictBtn.textContent = 'Predicting...';
        this.predictBtn.classList.add('loading');
        this.predictBtn.disabled = true;
    }
    
    hideLoading() {
        this.predictBtn.textContent = 'Predict';
        this.predictBtn.classList.remove('loading');
        this.predictBtn.disabled = false;
    }
    
    showPrediction(prediction, confidence) {
        const digitElement = this.predictionResult.querySelector('.digit-large');
        const confidenceElement = this.predictionResult.querySelector('.confidence');
        
        digitElement.textContent = prediction;
        confidenceElement.textContent = `Confidence: ${confidence.toFixed(1)}%`;
        
        // Add animation
        digitElement.style.transform = 'scale(1.2)';
        setTimeout(() => {
            digitElement.style.transform = 'scale(1)';
        }, 200);
    }
    
    showConfidenceScores(scores) {
        this.confidenceBars.innerHTML = '';
        
        // Sort scores by confidence
        const sortedScores = Object.entries(scores)
            .map(([digit, score]) => ({ digit, score }))
            .sort((a, b) => b.score - a.score);
        
        sortedScores.forEach(({ digit, score }) => {
            const barContainer = document.createElement('div');
            barContainer.className = 'confidence-bar';
            
            barContainer.innerHTML = `
                <div class="digit-label">${digit}</div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: ${score}%">
                        <div class="bar-percentage">${score.toFixed(1)}%</div>
                    </div>
                </div>
            `;
            
            this.confidenceBars.appendChild(barContainer);
        });
    }
    
    showError(message) {
        this.predictionResult.innerHTML = `
            <div class="digit-large" style="color: #dc3545;">!</div>
            <div class="confidence" style="color: #dc3545;">${message}</div>
        `;
        this.confidenceBars.innerHTML = '<div class="no-data">Error occurred</div>';
    }
    
    reset() {
        this.predictionResult.innerHTML = `
            <div class="digit-large">?</div>
            <div class="confidence">Draw a digit to see prediction</div>
        `;
        this.confidenceBars.innerHTML = '<div class="no-data">No prediction yet</div>';
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    const canvas = new DigitCanvas('drawCanvas');
    const api = new PredictionAPI();
    const ui = new UI();
    
    // Check API health on load
    api.checkHealth().then(health => {
        if (!health.model_loaded) {
            ui.showError('Model not loaded. Please check the server.');
        }
    });
    
    // Event listeners
    document.getElementById('predictBtn').addEventListener('click', async () => {
        const imageData = canvas.getImageData();
        
        // Check if canvas has content
        const hasContent = imageData.some(pixel => pixel < 250);
        if (!hasContent) {
            ui.showError('Please draw a digit first');
            return;
        }
        
        ui.showLoading();
        
        try {
            const result = await api.predict(imageData);
            ui.showPrediction(result.prediction, result.confidence);
            ui.showConfidenceScores(result.all_scores);
        } catch (error) {
            ui.showError('Failed to get prediction. Please try again.');
        } finally {
            ui.hideLoading();
        }
    });
    
    document.getElementById('clearBtn').addEventListener('click', () => {
        canvas.clear();
        ui.reset();
    });
});
