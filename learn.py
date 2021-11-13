from PIL import Image, ImageDraw
import torch
import numpy as np
import random
from torch import nn

IMAGE_SIZE = 64
OUTPUT_IMAGE_SIZE = 1024


def new_image_as_tensor(size):
    im = Image.new('L', (size,size))
    draw = ImageDraw.Draw(im)
    x = random.randrange(size-1)
    y = random.randrange(size-1)
    draw.ellipse((x,y,x+1,y+1),fill=(255))
    data = list(im.tobytes())
    data = torch.tensor(data)
    data = data.reshape(size,size).divide(255)
    return (data,torch.tensor((x/IMAGE_SIZE,y/IMAGE_SIZE)))

# im.save('image.png')
# q = im.tobytes()
# q = list(q)
# data = torch.tensor(q)
# data = data.reshape(256,256)
# print(data.shape)

data = new_image_as_tensor(IMAGE_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten(0)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(data, model, loss_fn, optimizer):
    size = len(data)
    model.train()
    for batch, (X, y) in data:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def rndclr():
    return (random.randrange(128)+127,random.randrange(128)+127,random.randrange(128)+127)
def test(data, model, loss_fn):
    size = len(data)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    im_out = Image.new('RGB', (OUTPUT_IMAGE_SIZE,OUTPUT_IMAGE_SIZE))
    draw = ImageDraw.Draw(im_out)
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = tuple(pred.multiply(OUTPUT_IMAGE_SIZE).tolist())
            y = tuple(y.multiply(OUTPUT_IMAGE_SIZE).tolist())
            draw.line([pred,y],fill=rndclr())
            
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    im_out.save("image.png")
    # correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 64

train_dataloader = []
test_dataloader = []
for i in range(512):
    item = new_image_as_tensor(IMAGE_SIZE)
    train_dataloader.append((i,item))
    item = new_image_as_tensor(IMAGE_SIZE)
    test_dataloader.append(item)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")