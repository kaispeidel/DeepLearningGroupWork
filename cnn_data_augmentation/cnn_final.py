import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pickle as pk
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add this code after importing libraries
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -- Data loading --------------------------------------------------------------
print('Loading training data')
with open('spectro_data_train_7979_filesls.pkl', 'rb') as f:
    train_data = pk.load(f)
print('Loading test data')
with open('spectro_data_test.pkl', 'rb') as f:
    test_data = pk.load(f)

# Custom dataset for labeled data
class AccentDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of (spectrogram, label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectro, label = self.data[idx]
        tensor = torch.from_numpy(spectro).float().unsqueeze(0)
        # Convert labels from range 1-5 to range 0-4
        return tensor, int(label) - 1

# Custom dataset for unlabeled test data
torch.manual_seed(0)
class SpectrogramDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of spectrogram numpy arrays

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectro = self.data[idx]
        tensor = torch.from_numpy(spectro).float().unsqueeze(0)
        return tensor

# Collate functions to pad variable-length spectrograms
def pad_collate(batch):
    """Pads a batch of (tensor, label) tuples to the same time dimension"""
    inputs, labels = zip(*batch)
    # find max time dim
    max_t = max(t.size(2) for t in inputs)
    padded = [F.pad(t, (0, max_t - t.size(2), 0, 0)) for t in inputs]
    inputs_tensor = torch.stack(padded)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return inputs_tensor, labels_tensor


def pad_collate_test(batch):
    """Pads a batch of tensors to the same time dimension"""
    inputs = batch
    max_t = max(t.size(2) for t in inputs)
    padded = [F.pad(t, (0, max_t - t.size(2), 0, 0)) for t in inputs]
    inputs_tensor = torch.stack(padded)
    return inputs_tensor

# Split training data into train and validation sets
val_size = int(0.2 * len(train_data))  # 20% for validation
train_size = len(train_data) - val_size
train_subset, val_subset = random_split(train_data, [train_size, val_size])

# DataLoaders with padding collate
trainset = AccentDataset(train_subset)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2, collate_fn=pad_collate)

# Create validation dataloader
valset = AccentDataset(val_subset)
valloader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=2, collate_fn=pad_collate)

testset = SpectrogramDataset(test_data)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2, collate_fn=pad_collate_test)

# -- Enhanced Model definition ---------------------------------------------------------
class EnhancedNet(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super().__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Adaptive pooling for variable size inputs
        self.adapt_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # First block with batch norm
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second block with batch norm
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third block with batch norm
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Adaptive pooling and flatten
        x = self.adapt_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
    
# Instantiate model, loss, optimizer
net = EnhancedNet(num_classes=5)  # Adjust based on your number of classes
net.to(device)
print(net)

# Better optimizer with weight decay for regularization
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

criterion = nn.CrossEntropyLoss()

# Add early stopping
early_stopping_patience = 10
best_val_accuracy = 0
early_stopping_counter = 0
best_model_path = 'best_model.pth'

# Function to calculate accuracy
def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training with early stopping
epochs = 100  # Maximum epochs
for epoch in range(epochs):
    # Training phase
    net.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(trainloader)
    
    # Validation phase
    val_accuracy = calculate_accuracy(net, valloader)
    train_accuracy = calculate_accuracy(net, trainloader)
    
    # Update learning rate based on validation performance
    scheduler.step(val_accuracy)
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

    # Early stopping check
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0
        # Save the best model
        torch.save(net.state_dict(), best_model_path)
        print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# -- Evaluation on test set ---------------------------------------------------
net.eval()
all_preds = []
with torch.no_grad():
    for inputs in testloader:
        inputs = inputs.to(device)  # Move inputs to device
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().tolist())  # Move predictions back to CPU

print(f'Generated predictions for {len(all_preds)} test samples.')

# Save predictions to a file
with open('predictions.txt', 'w') as f:
    for i, pred in enumerate(all_preds):
        f.write(f"{i},{pred}\n")

print(f'Predictions saved to predictions.txt')