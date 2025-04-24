import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from nn import Autoencoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tqdm import tqdm


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 128
learning_rate = 0.001
epochs = 100
latent_dim = 64

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FashionMNIST Dataset
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Model, Loss, Optimizer
model = Autoencoder(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Autoencoder Training Loop
for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0
    for images, _ in train_loader:
        # Forward pass
        outputs = model(images.to(device))
        loss = criterion(outputs, images.to(device))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Evaluate on Test Data
model.eval()
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images.to(device))
        break  # Display only one batch

# Visualize Original and Reconstructed Images

n = 10  # Number of images to display
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original images
    plt.subplot(2, n, i + 1)
    plt.imshow(images[i].squeeze().cpu(), cmap='gray')
    plt.axis('off')

    # Reconstructed images
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(outputs[i].squeeze().cpu(), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

#### SVC Training

classifier = SVC()

with torch.no_grad():
    for images, _ in test_loader:
        encode = model.encode(images.to(device))
        print(encode.shape)
        break  # Display only one batch

X_train = []
y_train = []

with torch.no_grad():
    for images, labels in tqdm(train_loader, desc='Extracting Train Features'):
        embeddings = model.encode(images.to(device))
        X_train.append(embeddings.detach().cpu().numpy())
        y_train.append(labels.cpu().numpy())

X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# Train the SVM classifier
classifier = SVC()
classifier.fit(X_train, y_train)

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        embeddings = model.encode(images.to(device))
        embeddings = embeddings.detach().cpu().numpy()
        labels = labels.numpy()
        y_pred = classifier.predict(embeddings)

        all_labels.extend(labels)
        all_preds.extend(y_pred)

    print(classification_report(all_labels, all_preds))
