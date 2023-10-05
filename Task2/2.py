import torch
import matplotlib.pyplot as plt
import numpy as np

x_train = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).reshape(-1, 2)
y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]]).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

     # Predictor
    def f(self, x):
        return self.ligma(self.logits(x))  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability
    
    def ligma(self, z):
        return torch.sigmoid(z)

model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.1)
# range 1000000 gir nice verdi
# range 100000 gir nice verdi mye raskere
for epoch in range(5000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step


# Visualize result
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0], label='Data')

grid_size = 10
x_vals = torch.linspace(torch.min(x_train[:, 0]), torch.max(x_train[:, 0]), grid_size)
y_vals = torch.linspace(torch.min(x_train[:, 1]), torch.max(x_train[:, 1]), grid_size)
X, Y = torch.meshgrid(x_vals, y_vals)

combined_grid = torch.stack((X.reshape(-1), Y.reshape(-1)), dim=1)

predictions = model.f(combined_grid).detach().reshape(grid_size, grid_size)
ax.plot_wireframe(X, Y, predictions, color='green', label='Model Prediction')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.legend()
plt.show()
