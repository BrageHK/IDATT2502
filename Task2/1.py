import torch
import matplotlib.pyplot as plt
import numpy as np

x_train = torch.tensor([1.0, 0.0]).reshape(-1, 1)
y_train = torch.tensor([0.0, 1.0]).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
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
print("w: ", model.W)

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
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x_values = np.linspace(-0.25, 1.25, 100)
y_values = [model.f(torch.tensor([[x]], dtype=torch.float32)).item() for x in x_values]
plt.plot(x_values, y_values, label='$y = f(x) = xW+b$')
plt.legend()
plt.show()
