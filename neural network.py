import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import networkx as nx # type: ignore
import matplotlib.cm as cm # type: ignore
import matplotlib.colors as mcolors # type: ignore

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def sigmoid_derivative(x):
    return x * (1 - x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Simple feedforward network with one hidden layer
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation="sigmoid"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.activation = activation

    def activate(self, x):
        return relu(x) if self.activation == "relu" else sigmoid(x)

    def activate_derivative(self, x):
        return relu_derivative(x) if self.activation == "relu" else sigmoid_derivative(x)
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activate(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.activate(self.final_input)
        return self.final_output

    def train(self, X, y, epochs=1000, lr=0.1):
        self.loss_history = []
        for _ in range(epochs):
            output = self.forward(X)
            error = y - output
            loss = np.mean(np.square(error))
            self.loss_history.append(loss)
            
            d_output = error * self.activate_derivative(output)
            error_hidden = d_output.dot(self.weights_hidden_output.T)
            d_hidden = error_hidden * self.activate_derivative(self.hidden_output)
            
            self.weights_hidden_output += self.hidden_output.T.dot(d_output) * lr
            self.bias_output += np.sum(d_output, axis=0, keepdims=True) * lr
            self.weights_input_hidden += X.T.dot(d_hidden) * lr
            self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr
    
    def visualize(self):
        G = nx.DiGraph()
        layers = [[f'Input {i+1}' for i in range(self.input_size)],
                  [f'Hidden {i+1}' for i in range(self.hidden_size)],
                  [f'Output {i+1}' for i in range(self.output_size)]]
        
        pos = {}
        for i, layer in enumerate(layers):
            for j, node in enumerate(layer):
                G.add_node(node, layer=i)
                pos[node] = (i, -j)
        
        weights = np.concatenate((self.weights_input_hidden.flatten(), self.weights_hidden_output.flatten()))
        norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
        cmap = cm.get_cmap('coolwarm')
        
        for i in range(len(layers) - 1):
            w_matrix = self.weights_input_hidden if i == 0 else self.weights_hidden_output
            for j, src in enumerate(layers[i]):
                for k, dest in enumerate(layers[i+1]):
                    weight = w_matrix[j, k]
                    G.add_edge(src, dest, weight=weight)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        edges = G.edges(data=True)
        colors = [cmap(norm(edge[2]['weight'])) for edge in edges]
        widths = [abs(edge[2]['weight']) * 3 for edge in edges]
        
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', edge_color=colors, width=widths, edge_cmap=cmap, ax=ax)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Weight Strength')
        plt.title("Neural Network - Visualization")
        plt.show()

# Test with data
np.random.seed(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR-problem

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation="relu")
nn.train(X, y)
nn.visualize()

print("Test:", nn.forward(X))
