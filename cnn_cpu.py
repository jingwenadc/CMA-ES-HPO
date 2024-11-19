import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Search range
CNN_LR = [0.0001, 0.1]
CNN_DROPOUT = [0.0, 0.5]
CNN_LAYERS = [1, 5]
CNN_NEURONS = [8, 512]
CNN_NUM_RUNS = 1


class CNN(nn.Module):
    def __init__(self, num_layers=3, num_neurons=128, dropout=0.5):
        super(CNN, self).__init__()
        layers = []
        input_channels = 3  # CIFAR-10 has 3 color channels
        for _ in range(num_layers):
            layers.append(nn.Conv2d(input_channels, num_neurons, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.MaxPool2d(kernel_size=2))
            input_channels = num_neurons
        self.features = nn.Sequential(*layers)

        # Dynamically calculate the size of the flattened feature map
        self._initialize_classifier(input_size=32, num_neurons=num_neurons)  # CIFAR-10 images are 32x32

    def _initialize_classifier(self, input_size, num_neurons):
        # Pass a dummy tensor through the features to calculate the flattened size
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.numel()  # Total number of elements in the flattened tensor
        self.classifier = nn.Linear(flattened_size, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    

# Early stopping criteria
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
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
    params[1]: dropout rate
    params[2]: number of layers
    params[3]: number of neurons per layer
    """
    learning_rate = params[0]
    dropout = params[1]
    num_layers = int(round(params[2]))
    num_neurons = int(round(params[3]))

    # Build and train model
    model = CNN(num_layers=num_layers, num_neurons=num_neurons, dropout=dropout)
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


def data_loader():
    # Load CIFAR-10 Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Step 2: Download the CIFAR-10 dataset and apply transformations
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Step 3: Split the dataset into training and validation sets (e.g., 80% training, 20% validation)
    train_size = int(0.8 * len(full_dataset))  # 80% for training
    val_size = len(full_dataset) - train_size  # 20% for validation
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader