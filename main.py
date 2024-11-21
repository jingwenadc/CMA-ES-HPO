import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from cma import CMAEvolutionStrategy
from torch.utils.data import DataLoader, random_split


# 1. Load CIFAR-10 Dataset
def load_data(batch_size=128):
    # Data preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(32, padding=4),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize to mean/std of CIFAR-10
    ])

    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Step 3: Split the dataset into training and validation sets (e.g., 70% training, 30% validation)
    train_size = int(0.7 * len(full_dataset))  # 70% for training
    val_size = len(full_dataset) - train_size  # 30% for validation
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# 2. Define CNN Architecture
class CNN(nn.Module):
    def __init__(self, kernel_size=3, dropout=0.5):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Dynamically calculate the size of the flattened feature map
        self._initialize_classifier(input_size=32)

    def _initialize_classifier(self, input_size):
        # Pass a dummy tensor through the features to calculate the flattened size
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.numel()  # Total number of elements in the flattened tensor
        self.classifier = nn.Linear(flattened_size, 512)  # Match the input size of the classifier
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

# Early stopping criteria
class EarlyStopping:
    def __init__(self, patience=3, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.counter = 0
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.save_checkpoint(model)
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


# Objective Function
def objective_function(params):
    """
    Objective function for CMA-ES:
    params[0]: learning rate
    params[1]: kernel size
    params[2]: dropout rate
    """
    learning_rate = params[0]
    kernel_size = int(params[1])
    dropout = params[2]

    # Build and train model
    model = CNN(kernel_size=kernel_size, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping()

    # Train for multiple epochs
    for epoch in range(100):  # You can adjust this number as needed
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs, labels
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate on validation data (or use test data as a proxy)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:  # Assuming you have a validation set
                inputs, labels = inputs, labels
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total

        # Check for early stopping
        if early_stopping(val_accuracy, model):
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs, labels
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return -correct / total


if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_data()


    # CMA-ES Setup
    initial_guess = [0.001, 3, 0.5]  # [learning_rate, kernel_size, dropout]
    sigma = 0.5
    bounds = [[0.0001, 3, 0.0],  # Lower bounds: learning_rate, dropout, num_layers, num_neurons
            [0.1, 7, 0.7]]  # Upper bounds: learning_rate, dropout, num_layers, num_neurons

    options = {
        'popsize': 5,       # Population size
        'tolfun': 1e-6       # Stop if the function value difference is below this
    }
    es = CMAEvolutionStrategy(initial_guess, sigma, {'bounds': bounds}, options)
    es.optimize(objective_function, iterations=10)

    # Best hyperparameters
    best_params = es.result.xbest
    print("Best Parameters:", best_params)

    print(es.result)