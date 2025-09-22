import numpy as np
import pickle
import gzip
import os
from urllib.request import urlretrieve

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=100, batch_size=32):
        m = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, m, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                output = self.forward(batch_X)
                self.backward(batch_X, batch_y, output)
            
            if epoch % 10 == 0:
                predictions = self.predict(X)
                accuracy = np.mean(predictions == np.argmax(y, axis=1))
                print(f"Epoch {epoch}, Accuracy: {accuracy:.4f}")
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X)
    
    def save_model(self, filename):
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            model_data['input_size'],
            model_data['hidden_size'],
            model_data['output_size'],
            model_data['learning_rate']
        )
        model.W1 = model_data['W1']
        model.b1 = model_data['b1']
        model.W2 = model_data['W2']
        model.b2 = model_data['b2']
        return model

def download_mnist():
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    for file in files:
        if not os.path.exists(file):
            print(f"Downloading {file}...")
            try:
                urlretrieve(base_url + file, file)
            except:
                print(f"Failed to download {file}, generating synthetic data instead...")
                generate_synthetic_mnist()
                return

def generate_synthetic_mnist():
    """Generate synthetic MNIST-like data for demonstration"""
    print("Generating synthetic MNIST data...")
    
    np.random.seed(42)
    
    # Generate training data
    train_images = []
    train_labels = []
    
    for digit in range(10):
        for _ in range(1000):  # 1000 samples per digit
            # Create a simple pattern for each digit
            image = np.zeros((28, 28))
            
            if digit == 0:
                # Circle pattern
                center = (14, 14)
                for i in range(28):
                    for j in range(28):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        if 8 <= dist <= 10:
                            image[i, j] = 1
            elif digit == 1:
                # Vertical line
                image[6:22, 12:16] = 1
            elif digit == 2:
                # Top horizontal, diagonal, bottom horizontal
                image[6, 8:20] = 1
                image[6:14, 18:20] = 1
                image[14, 8:20] = 1
                image[14:22, 8:10] = 1
                image[22, 8:20] = 1
            elif digit == 3:
                # Two vertical lines with horizontal connections
                image[6:22, 8:10] = 1
                image[6:22, 18:20] = 1
                image[6, 8:20] = 1
                image[14, 8:20] = 1
                image[22, 8:20] = 1
            elif digit == 4:
                # Vertical line and horizontal line
                image[6:14, 8:10] = 1
                image[14, 8:20] = 1
                image[14:22, 18:20] = 1
            elif digit == 5:
                # Horizontal lines with vertical connections
                image[6, 8:20] = 1
                image[6:14, 8:10] = 1
                image[14, 8:20] = 1
                image[14:22, 18:20] = 1
                image[22, 8:20] = 1
            elif digit == 6:
                # Rectangle with opening
                image[6:22, 8:10] = 1
                image[6, 8:20] = 1
                image[14, 8:20] = 1
                image[22, 8:20] = 1
                image[14:22, 18:20] = 1
            elif digit == 7:
                # Top horizontal and diagonal
                image[6, 8:20] = 1
                image[6:22, 18:20] = 1
            elif digit == 8:
                # Two rectangles
                image[6:14, 8:10] = 1
                image[6:14, 18:20] = 1
                image[6, 8:20] = 1
                image[14, 8:20] = 1
                image[14:22, 8:10] = 1
                image[14:22, 18:20] = 1
                image[22, 8:20] = 1
            elif digit == 9:
                # Rectangle with opening at bottom
                image[6:14, 8:10] = 1
                image[6:14, 18:20] = 1
                image[6, 8:20] = 1
                image[14, 8:20] = 1
                image[14:22, 18:20] = 1
            
            # Add some noise
            noise = np.random.normal(0, 0.1, (28, 28))
            image = np.clip(image + noise, 0, 1)
            
            train_images.append(image.flatten())
            train_labels.append(digit)
    
    # Generate test data (smaller set)
    test_images = []
    test_labels = []
    
    for digit in range(10):
        for _ in range(200):  # 200 samples per digit
            # Similar pattern generation as above
            image = np.zeros((28, 28))
            
            if digit == 0:
                center = (14, 14)
                for i in range(28):
                    for j in range(28):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        if 8 <= dist <= 10:
                            image[i, j] = 1
            elif digit == 1:
                image[6:22, 12:16] = 1
            elif digit == 2:
                image[6, 8:20] = 1
                image[6:14, 18:20] = 1
                image[14, 8:20] = 1
                image[14:22, 8:10] = 1
                image[22, 8:20] = 1
            elif digit == 3:
                image[6:22, 8:10] = 1
                image[6:22, 18:20] = 1
                image[6, 8:20] = 1
                image[14, 8:20] = 1
                image[22, 8:20] = 1
            elif digit == 4:
                image[6:14, 8:10] = 1
                image[14, 8:20] = 1
                image[14:22, 18:20] = 1
            elif digit == 5:
                image[6, 8:20] = 1
                image[6:14, 8:10] = 1
                image[14, 8:20] = 1
                image[14:22, 18:20] = 1
                image[22, 8:20] = 1
            elif digit == 6:
                image[6:22, 8:10] = 1
                image[6, 8:20] = 1
                image[14, 8:20] = 1
                image[22, 8:20] = 1
                image[14:22, 18:20] = 1
            elif digit == 7:
                image[6, 8:20] = 1
                image[6:22, 18:20] = 1
            elif digit == 8:
                image[6:14, 8:10] = 1
                image[6:14, 18:20] = 1
                image[6, 8:20] = 1
                image[14, 8:20] = 1
                image[14:22, 8:10] = 1
                image[14:22, 18:20] = 1
                image[22, 8:20] = 1
            elif digit == 9:
                image[6:14, 8:10] = 1
                image[6:14, 18:20] = 1
                image[6, 8:20] = 1
                image[14, 8:20] = 1
                image[14:22, 18:20] = 1
            
            noise = np.random.normal(0, 0.1, (28, 28))
            image = np.clip(image + noise, 0, 1)
            
            test_images.append(image.flatten())
            test_labels.append(digit)
    
    # Convert to numpy arrays
    train_images = np.array(train_images, dtype=np.float32)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images, dtype=np.float32)
    test_labels = np.array(test_labels)
    
    # One-hot encode labels
    train_labels_onehot = np.eye(10)[train_labels]
    test_labels_onehot = np.eye(10)[test_labels]
    
    return train_images, train_labels_onehot, test_images, test_labels_onehot

def load_mnist():
    # Try to download real MNIST first
    try:
        download_mnist()
        
        def read_images(filename):
            with gzip.open(filename, 'rb') as f:
                f.read(16)  # Skip header
                buf = f.read()
                data = np.frombuffer(buf, dtype=np.uint8)
                return data.reshape(-1, 28, 28)
        
        def read_labels(filename):
            with gzip.open(filename, 'rb') as f:
                f.read(8)  # Skip header
                buf = f.read()
                return np.frombuffer(buf, dtype=np.uint8)
        
        train_images = read_images('train-images-idx3-ubyte.gz')
        train_labels = read_labels('train-labels-idx1-ubyte.gz')
        test_images = read_images('t10k-images-idx3-ubyte.gz')
        test_labels = read_labels('t10k-labels-idx1-ubyte.gz')
        
        # Normalize and flatten images
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
        train_images = train_images.reshape(-1, 784)
        test_images = test_images.reshape(-1, 784)
        
        # One-hot encode labels
        train_labels_onehot = np.eye(10)[train_labels]
        test_labels_onehot = np.eye(10)[test_labels]
        
        return train_images, train_labels_onehot, test_images, test_labels_onehot
        
    except:
        # Fall back to synthetic data
        print("Using synthetic MNIST data...")
        return generate_synthetic_mnist()
