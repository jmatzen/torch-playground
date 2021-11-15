from PIL import Image, ImageDraw
import torch
import numpy as np
import random
from torch import nn, Tensor
import math
import ffmpeg

IMAGE_SIZE = 64
OUTPUT_IMAGE_SIZE = 1024
DOT_SIZE=4


vid = (ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(OUTPUT_IMAGE_SIZE,OUTPUT_IMAGE_SIZE))
        .output("video.mp4", pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
def rndclr():
    return (random.randrange(128)+127,random.randrange(128)+127,random.randrange(128)+127)

def new_image_as_tensor(size):
    im = Image.new('L', (size,size))
    draw = ImageDraw.Draw(im)
    x = random.randrange(size-1)
    y = random.randrange(size-1)
    draw.ellipse((x,y,x+1,y+1),fill=(255))
    data = list(im.tobytes())
    data = torch.tensor(data).to(device)
    data = data.reshape(size,size).divide(255)
    return (data,torch.tensor((x/IMAGE_SIZE,y/IMAGE_SIZE)),rndclr())

# im.save('image.png')
# q = im.tobytes()
# q = list(q)
# data = torch.tensor(q)
# data = data.reshape(256,256)
# print(data.shape)

data = new_image_as_tensor(IMAGE_SIZE)



# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten(0)

        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 2),
            # nn.ReLU(),
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# class Loss():
#     def __init__(self):
#         self.mse = nn.MSELoss()

#     def __call__(self, input: torch.Tensor, target: torch.Tensor)->torch.Tensor:
#         err =  self.mse.forward(input,target)
#         return math.inf if err > 0.01 else err



# loss_fn = Loss()

class Loss(nn.MSELoss):
    def __init__(self):
        super(nn.MSELoss,self).__init__()
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = nn.MSELoss.forward(self, input, target)
        # loss.pow(.01)
        return loss

loss_fn = Loss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

def train(data, model, loss_fn, optimizer):
    size = len(data)
    model.train()
    for batch, (X, y, color) in data:
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


def test(data, model, loss_fn, train_data):
    size = len(data)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    im_out = Image.new('RGB', (OUTPUT_IMAGE_SIZE,OUTPUT_IMAGE_SIZE))
    draw = ImageDraw.Draw(im_out)
    with torch.no_grad():
        for X, y, color in data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = tuple(pred.multiply(OUTPUT_IMAGE_SIZE).tolist())
            y = tuple(y.multiply(OUTPUT_IMAGE_SIZE).tolist())
            # draw.line([pred,y],fill=rndclr())
            draw.ellipse([pred[0]-DOT_SIZE,pred[1]-DOT_SIZE,pred[0]+DOT_SIZE,pred[1]+DOT_SIZE], fill=color)
            draw.ellipse([y[0]-DOT_SIZE,y[1]-DOT_SIZE,y[0]+DOT_SIZE,y[1]+DOT_SIZE], fill=color)
            draw.line([pred[0], pred[1], y[0], y[1]], fill=color)
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches

    for batch, (X, y, color) in train_data:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        pred = tuple(pred.multiply(OUTPUT_IMAGE_SIZE).tolist())
        y = tuple(y.multiply(OUTPUT_IMAGE_SIZE).tolist())
        draw.ellipse([pred[0],pred[1],pred[0]+1,pred[1]+1], fill=color)
        draw.ellipse([y[0],y[1],y[0]+1,y[1]+1], fill=color)
        draw.line([pred[0], pred[1], y[0], y[1]], fill=color)

    im_out.save("image.png")
    vid.stdin.write(im_out.tobytes())
    
    # correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 512




for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_dataloader = []
    test_dataloader = []
    for i in range(256):
        item = new_image_as_tensor(IMAGE_SIZE)
        train_dataloader.append((i,item))
    for i in range(8):
        item = new_image_as_tensor(IMAGE_SIZE)
        test_dataloader.append(item)
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn, train_dataloader)
print("Done!")