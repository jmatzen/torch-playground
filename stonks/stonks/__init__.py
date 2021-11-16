from torch import nn, Tensor

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten(0)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5*5, 3*5),
            nn.Linear(3*5, 5),
            # nn.ReLU(),
            # nn.Linear(8*6, 4*6),
            # # nn.ReLU(),
            # nn.Linear(4*6, 6),
            # # nn.ReLU(),
            # nn.Linear(6, 1),
            # # nn.ReLU(),

            # nn.Linear(16*5,4*5),
            # nn.Linear(4*5,2*5),
            # nn.Linear(2*5,5)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def normalize_col(rows,n):
    min = 99999999999
    max = 0
    for r in rows:
        min = r[n] if r[n] < min else min
        max = r[n] if r[n] > max else max
    for r in rows:
        r[n] = ((r[n]-min) / (max-min)) * 2.0 - 1.0
    return (min,max)
    

def normalize(rows):
    scale=[]
    scale.append(normalize_col(rows,0))
    scale.append(normalize_col(rows,1))
    scale.append(normalize_col(rows,2))
    scale.append(normalize_col(rows,3))
    scale.append(normalize_col(rows,4))
    return scale
