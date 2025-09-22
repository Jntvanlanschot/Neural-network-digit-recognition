from neural_network import NeuralNetwork, load_mnist
import numpy as np

def main():
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train neural network
    print("\nCreating neural network...")
    nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10, learning_rate=0.01)
    
    print("Training neural network...")
    nn.train(X_train, y_train, epochs=50, batch_size=32)
    
    # Test the model
    print("\nTesting model...")
    test_predictions = nn.predict(X_test)
    test_accuracy = np.mean(test_predictions == np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save the trained model
    print("\nSaving model...")
    nn.save_model('model.pkl')
    print("Model saved successfully!")
    
    # Test loading the model
    print("\nTesting model loading...")
    loaded_nn = NeuralNetwork.load_model('model.pkl')
    loaded_predictions = loaded_nn.predict(X_test[:100])
    loaded_accuracy = np.mean(loaded_predictions == np.argmax(y_test[:100], axis=1))
    print(f"Loaded model accuracy (first 100 samples): {loaded_accuracy:.4f}")

if __name__ == "__main__":
    main()
