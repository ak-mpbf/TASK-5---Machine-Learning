import numpy as np
import argparse
import pickle
import os
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.5, help='momentum')
parser.add_argument('--num_hidden', type=int, default=2, help='number of hidden layers')
parser.add_argument('--sizes', type=str, default='200,100', help='comma separated list of hidden layer sizes')
parser.add_argument('--activation', type=str, default='sigmoid', help='activation function')
parser.add_argument('--loss', type=str, default='sq', help='loss function')
parser.add_argument('--opt', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--batch_size', type=int, default=20, help='batch size')
parser.add_argument('--anneal', type=bool, default=True, help='halve learning rate if validation loss decreases')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='directory to save the model')
parser.add_argument('--expt_dir', type=str, default='./logs', help='directory to save log files')
parser.add_argument('--train', type=str, default='./cifar10/cifar-10-batches-py/data_batch_1', help='path to training dataset')
parser.add_argument('--test', type=str, default='./cifar10/cifar-10-batches-py/test_batch', help='path to test dataset')

args = parser.parse_args()

def load_data(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def load_cifar10(train_path, test_path):
    train_data = []
    for i in range(1, 6):
        path = f'./cifar10/cifar-10-batches-py/data_batch_{i}'
        train_data.append(load_data(path))
    test_data = load_data(test_path)
    train_images = np.concatenate([data[b'data'] for data in train_data], axis=0)
    train_labels = np.concatenate([data[b'labels'] for data in train_data], axis=0)
    test_images = test_data[b'data']
    test_labels = test_data[b'labels']
    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_cifar10(args.train, args.test)

class NeuralNetwork:
    def __init__(self, sizes, activation):
        self.sizes = sizes
        self.activation = activation
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i+1], sizes[i]) * np.sqrt(2 / sizes[i]))
            self.biases.append(np.zeros((sizes[i+1], 1)))

    def forward(self, x):
        self.z_values = []
        self.activations = []

        # Perform forward pass
        activation = x
        self.activations.append(activation)

        for i in range(len(self.sizes) - 1):
            z = np.dot(self.weights[i], activation) + self.biases[i]

            if self.activation == 'sigmoid':
                activation = sigmoid(z)
            elif self.activation == 'tanh':
                activation = tanh(z)
            else:
                raise ValueError('Invalid activation function.')

            self.z_values.append(z)
            self.activations.append(activation)

    def backward(self, x, y):
        m = x.shape[1]
        dz_values = []
        dw_values = []
        db_values = []

        # Compute gradients for each layer using backpropagation
        if self.activation == 'sigmoid':
            derivative = sigmoid_derivative
        elif self.activation == 'tanh':
            derivative = tanh_derivative

        dz = self.activations[-1] - y
        dw = np.dot(dz, self.activations[-2].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        dz_values.append(dz)
        dw_values.append(dw)
        db_values.append(db)

        for i in range(len(self.sizes) - 2, 0, -1):
            dz = np.dot(self.weights[i].T, dz_values[-1]) * derivative(self.z_values[i-1])
            dw = np.dot(dz, self.activations[i-1].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            dz_values.append(dz)
            dw_values.append(dw)
            db_values.append(db)

        return dz_values, dw_values[::-1], db_values[::-1]

    def update_parameters(self, dw_values, db_values, lr):
        # Update weights and biases using gradient descent
        for i in range(len(self.sizes) - 1):
            self.weights[i] -= lr * dw_values[i]
            self.biases[i] -= lr * db_values[i]

    def train(self, x, y, lr):
        # Perform forward pass, backward pass, and update parameters
        self.forward(x)
        dz_values, dw_values, db_values = self.backward(x, y)
        self.update_parameters(dw_values, db_values, lr)

    def predict(self, x):
        # Make predictions by performing a forward pass
        self.forward(x)
        return np.argmax(self.activations[-1], axis=0)

    def compute_loss(self, x, y):
        # Compute loss using the specified loss function
        self.forward(x)
        m = x.shape[1]

        if args.loss == 'sq':
            loss = np.sum((self.activations[-1] - y)**2) / (2 * m)
        elif args.loss == 'ce':
            loss = -np.sum(y * np.log(abs(self.activations[-1] + 1e-8))) / m
        else:
            raise ValueError('Invalid loss function.')

        return loss
    
    def error_count(self, x, y):
        count = 0
        for i in range(0, len(y.T)):
            if self.predict(x.T[i])[0] != np.argmax(y.T[i], axis=0):
                count += 1
        return count
    

# Define the optimization algorithm
def gradient_descent(nn, x_train, y_train, x_val, y_val, lr, num_epochs, batch_size, anneal):
    m = x_train.shape[1]
    train_losses = []
    val_losses = []

    os.makedirs(args.expt_dir, exist_ok=True)
    train_log = open(args.expt_dir+"/log train.txt",'w+')
    val_log = open(args.expt_dir+"/log val.txt",'w+')
    epoch=0
    while epoch < num_epochs:
        if anneal and len(val_losses) > 1 and val_losses[-1] < val_losses[-2]:
            lr /= 2
            epoch -= 1

        for batch_start in range(0, m, batch_size):
            batch_end = min(batch_start + batch_size, m)
            x_batch = x_train[:, batch_start:batch_end]
            y_batch = y_train[:, batch_start:batch_end]

            nn.train(x_batch, y_batch, lr)
        for step in range(100, m, 100):
            x_batch = x_train[:, step-100:step]
            y_batch = y_train[:, step-100:step]
            train_loss = nn.compute_loss(x_batch, y_batch)
            train_error = nn.error_count(x_batch, y_batch)
            train_log.write(f'Epoch {epoch}, Step {step}, Loss: {train_loss:.4f}, Error: {train_error}, lr: {lr}\n')

        for step in range(100, x_val.shape[1], 100):
            x_batch = x_val[:, step-100:step]
            y_batch = y_val[:, step-100:step]
            val_loss = nn.compute_loss(x_batch, y_batch)
            val_error = nn.error_count(x_batch, y_batch)
            val_log.write(f'Epoch {epoch}, Step {step}, Loss: {val_loss:.4f}, Error: {val_error}, lr: {lr}\n')

        train_loss = nn.compute_loss(x_train, y_train)
        val_loss = nn.compute_loss(x_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        epoch+=1

    return train_losses, val_losses

# Initialize and train the neural network
sizes = [32*32*3] + [int(size) for size in args.sizes.split(',')] + [10]
activation = args.activation

nn = NeuralNetwork(sizes, activation)

os.makedirs(args.save_dir, exist_ok=True)
model_path = os.path.join(args.save_dir, 'model.pkl')

x_train = train_images.reshape(train_images.shape[0], -1).T / 255.0
y_train = np.eye(10)[train_labels].T
x_val = test_images.reshape(test_images.shape[0], -1).T / 255.0
y_val = np.eye(10)[test_labels].T

train_losses, val_losses = gradient_descent(nn, x_train, y_train, x_val, y_val,
                                            args.lr, 3, args.batch_size, args.anneal)

# Save the trained model

with open(model_path, 'wb') as f:
    pickle.dump(nn, f)

print(f'Trained model saved at: {model_path}')

number_to_label=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

predicted_labels=nn.predict(x_val)
predicted_labels=[number_to_label[i] for i in predicted_labels]
actual_answer=[number_to_label[i] for i in test_labels]

count1=0
count2=0
for i in range(len(predicted_labels)):
        if predicted_labels[i] != actual_answer[i]:
            count2+=1
        else :
            count2+=1
            count1+=1
print("% of correct predictions :-")
print(count1*100/count2)

submission_data = pd.DataFrame({'id': range(1, len(predicted_labels) + 1),'label': predicted_labels,'asli': actual_answer})
# Save the DataFrame as a CSV file
submission_data.to_csv(args.expt_dir+"/submission.csv", index=False)


#print(sum([1 if nn.predict(x_val[i])==np.argmax(y_val[i]) else 0 for i in range(10000)]))
