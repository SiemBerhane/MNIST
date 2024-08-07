import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

training_dataset = datasets.MNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor()
)

test_dataset = datasets.MNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential (
            nn.Linear(28*28, 30),
            nn.Sigmoid(),
            nn.Linear(30, 30),
            nn.Sigmoid(),
            nn.Linear(30, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer, lambda_reg):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        l2_norm /= 2*sum(p.numel() for p in model.parameters())
        loss = loss_fn(pred, y) + lambda_reg*l2_norm

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0: # Checks the progress every 100 batches
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    print(f"Training Accuracy: {correct*100:>.2f}%") 
    return correct


def test_loop(dataloader, model, loss_fn, lambda_reg):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            l2_norm /= 2*sum(p.numel() for p in model.parameters())
            test_loss += loss_fn(pred, y).item() + lambda_reg*l2_norm
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    return correct

learning_rate = 0.5
lambda_reg = 3
batch_size = 10
epochs = 30

train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model = NeuralNetwork()

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_correct = []
test_correct = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_correct.append(train_loop(train_dataloader, model, loss_fn, optimizer, lambda_reg))
    test_correct.append(test_loop(test_dataloader, model, loss_fn, lambda_reg))
print("Done!")


# to visualise overfitting
plt.plot(train_correct, label='training')
plt.plot(test_correct, label='testing')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy Percantage')

plt.legend()
plt.show()
