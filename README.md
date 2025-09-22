# Neural Network Digit Recognition

A complete neural network implementation from scratch using only Python and NumPy, trained on MNIST dataset and deployed as an interactive web application.

## Features

- **Neural Network from Scratch**: Built using only Python and NumPy (no TensorFlow/PyTorch)
- **MNIST Training**: Trained on handwritten digit dataset
- **Interactive Web UI**: Draw digits and get real-time predictions
- **Confidence Scores**: See prediction probabilities for all digits
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
├── backend/
│   ├── neural_network.py      # Neural network implementation
│   ├── train_model.py         # Training script
│   ├── model.pkl             # Trained model (generated)
│   └── requirements.txt      # Python dependencies
├── api/
│   ├── app.py                # Flask API server
│   └── requirements.txt     # API dependencies
├── frontend/
│   ├── index.html           # Main HTML page
│   ├── style.css            # Styling
│   └── script.js            # Frontend logic
└── README.md               # This file
```

## Local Development

### 1. Train the Model

```bash
cd backend
pip install -r requirements.txt
python train_model.py
```

This will:
- Download MNIST dataset
- Train the neural network
- Save the model as `model.pkl`

### 2. Run the API Server

```bash
cd api
pip install -r requirements.txt
python app.py
```

The API will be available at `http://localhost:5000`

### 3. Open the Frontend

Open `frontend/index.html` in your browser or serve it through the Flask app.

## Deployment on Vercel

### 1. Prepare for Deployment

Create a `vercel.json` file in the root directory:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "api/app.py"
    },
    {
      "src": "/(.*)",
      "dest": "frontend/$1"
    }
  ]
}
```

### 2. GitHub Setup

1. Create a new GitHub repository
2. Push your code to GitHub
3. Make sure `model.pkl` is included in the repository

### 3. Deploy to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repository
3. Vercel will automatically detect the Python app
4. Deploy!

### 4. Environment Variables (if needed)

If you need any environment variables, add them in the Vercel dashboard under Project Settings > Environment Variables.

## API Endpoints

- `GET /` - Serves the frontend
- `POST /predict` - Predicts digit from image data
- `GET /health` - Health check endpoint

## Model Architecture

- **Input Layer**: 784 neurons (28x28 pixels)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation
- **Training**: Mini-batch gradient descent with 50 epochs

## Performance

- **Training Accuracy**: ~95-98%
- **Test Accuracy**: ~94-97%
- **Inference Time**: <100ms per prediction

## Future Improvements

- Add more hidden layers
- Implement dropout for regularization
- Add data augmentation
- Implement convolutional layers
- Add model versioning
- Implement batch prediction API
- Add model retraining capabilities
