import torch
import matplotlib.pyplot as plt
import numpy as np

x_train = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

class LinearRegressionModel:

    def __init__(self, W1, b1, W2, b2):
        # Model variables
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def logits(self, x):
        hidden = self.ligma(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2

     # Predictor
    def f(self, x):
        return self.ligma(self.logits(x)) 

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)
    
    # lol
    def ligma(self, z):
        return torch.sigmoid(z)


W1 = torch.tensor([[10., -10.], [10., -10.]])
b1 = torch.tensor([[-5., 15.]], requires_grad=True)
W2 = torch.tensor([[10.], [10.]], requires_grad=True)
b2 = torch.tensor([[-15.]], requires_grad=True)

W1rand = 2 * torch.rand(2, 2) - 1
b1rand = 2 * torch.rand(1, 2) - 1
W2rand = 2 * torch.rand(2, 1) - 1
b2rand = 2 * torch.rand(1, 1) - 1
W1rand.requires_grad = True
b1rand.requires_grad = True
W2rand.requires_grad = True
b2rand.requires_grad = True


model1 = LinearRegressionModel(W1, b1, W2, b2)
model2 = LinearRegressionModel(W1rand, b1rand, W2rand, b2rand)


optimizer1 = torch.optim.SGD([model1.W1, model1.b1, model1.W2, model1.b2], 0.1)
optimizer2 = torch.optim.SGD([model2.W1, model2.b1, model2.W2, model2.b2], 0.1)
for epoch in range(10000):
    model1.loss(x_train, y_train).backward()
    model2.loss(x_train, y_train).backward() 
    optimizer1.step()
    optimizer2.step()

    optimizer1.zero_grad()
    optimizer2.zero_grad()


# Visualize result
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0], label='Data')

grid_size = 10
x_vals = torch.linspace(torch.min(x_train[:, 0]), torch.max(x_train[:, 0]), grid_size)
y_vals = torch.linspace(torch.min(x_train[:, 1]), torch.max(x_train[:, 1]), grid_size)
X, Y = torch.meshgrid(x_vals, y_vals)

combined_grid = torch.stack((X.reshape(-1), Y.reshape(-1)), dim=1)

predictions = model1.f(combined_grid).detach().reshape(grid_size, grid_size)
ax.plot_wireframe(X, Y, predictions, color='green', label='Model Prediction')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0], label='Data')
predictions2 = model2.f(combined_grid).detach().reshape(grid_size, grid_size)
ax2.plot_wireframe(X, Y, predictions2, color='blue', label='Model Prediction')

plt.legend()
plt.show()
