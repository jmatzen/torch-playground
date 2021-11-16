from PIL import Image, ImageDraw
import torch
import numpy as np
import random
from torch import nn, Tensor
import math
from stonks import *

device = "cpu"# "cuda" if torch.cuda.is_available() else "cpu"
print(device)


model = NeuralNetwork().to(device)
print(model)
lossfn = nn.HuberLoss()
#lossfn = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adagrad(model.parameters(),lr=0.01)


def train(data, model, loss_fn, optimizer):
    size = len(data)
    model.train()
    test_loss = 0
    for (X, y, _) in data:
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), 1 * len(X)
        test_loss += loss / len(data)

def test(data, model, loss_fn):
    size = len(data)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X, y, _) in data:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            y = y.tolist()[3]
            pred = pred.tolist()[3]
            x = X.tolist()[-1:][0][3]
            if ((y<x and pred<x) or (y>x and pred>x)):
                correct = correct + 1
                # print(f"x={x} y={y} pred={pred}")
    test_loss /= num_batches
    correct = correct / num_batches

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

data=[]
with open("data.txt") as f:
    for line in f:
        q = eval(line)
        rawdata = torch.tensor(q['data']).to(device)
        target = torch.tensor(q['target']).to(device)
        data.append((rawdata,target,q))
print(len(data))
n = math.floor(len(data)*.1)
test_data = []
train_data = data
for i in range(n):
    idx = random.randint(0,i)
    test_data.append(train_data[idx])
    train_data[idx] = train_data.pop()
print(f'test_data.size={len(test_data)} train_data.size={len(train_data)}')

for t in range(16):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_data, model, lossfn, optimizer)
    test(test_data, model, lossfn)



# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # x = model.state_dict()[param_tensor].tolist()
    # print(x)

torch.save(model, "model.bin")
