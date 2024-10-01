import numpy as np

class NeuralNet:
    def __init__(self) -> None:
        np.random.seed(1)
        
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, y):
        return y * (1-y)
    
    def predict(self, inputs):
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))
    
    def train(self, inputs, outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.predict(inputs)
            
            error = outputs - output
            
            factor = np.dot(inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += factor
            
if __name__ == "__main__":
    neural_net = NeuralNet()
    
    inputs = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    outputs = np.array([[0, 0, 1]]).T
    
    neural_net.train(inputs, outputs, 100000)
    
    print(neural_net.synaptic_weights)
    print("\n\n")
    print(neural_net.predict([0, 1, 1]))
    print(neural_net.predict([1, 0, 0]))
    print(neural_net.predict([1, 0, 1]))
