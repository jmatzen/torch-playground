from PIL import Image, ImageDraw
import torch
import numpy as np
import random
from torch import nn, Tensor
import math


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten(0)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16*6, 8*6),
            nn.ReLU(),
            nn.Linear(8*6, 4*6),
            nn.ReLU(),
            nn.Linear(4*6, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.ReLU(),
            # nn.Linear(32, 1),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
lossfn = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adagrad(model.parameters(),lr=0.03)


def train(data, model, loss_fn, optimizer):
    size = len(data)
    model.train()
    test_loss = 0
    for (X, y) in data:
        # X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), 1 * len(X)
        test_loss += loss / len(data)
    # print(f"loss: {test_loss:>7f}")


def test(data, model, loss_fn):
    size = len(data)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            y = y.item()
            pred = pred.item()
            if ((y<1 and pred<1) or (y>1 and pred>1)):
                correct = correct + 1
    test_loss /= num_batches
    correct = correct / num_batches

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

data=[]
with open("data.txt") as f:
    for line in f:
        q = eval(line)
        a = torch.tensor(q[0]).to(device)
        b = torch.tensor(q[1]).to(device)
        data.append((a,b))
print(len(data))
n = math.floor(len(data)*.9)
train_data = data[n:]
test_data = data[:n]

for t in range(512):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_data, model, lossfn, optimizer)
    test(test_data, model, lossfn)

