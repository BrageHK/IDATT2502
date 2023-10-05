import torch
import torchvision
import matplotlib.pyplot as plt

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output


class SoftmaxRegressionModel:
  def __init__(self):
    self.W = torch.randn(784, 10, requires_grad=True)
    self.b = torch.zeros(10, requires_grad=True)

  def logits(self, x):
    return x @ self.W + self.b

  def f(self, x):
    return torch.nn.functional.softmax(self.logits(x), dim=1)
  
  def loss(self, x, y):
    return torch.nn.functional.cross_entropy(self.logits(x), y)

  def accuracy(self, x, y):
    return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())
  
model = SoftmaxRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in range(300):
  model.loss(x_train, y_train).backward()
  optimizer.step()
  optimizer.zero_grad()

print("Accuracy on test set:", model.accuracy(x_test, y_test).item())

for i in range(10):
  name = 'W' + str(i) + '.png'
  plt.imsave(name, model.W[:, i].detach().numpy().reshape(28, 28))