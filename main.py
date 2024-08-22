import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import io

# Loading the training and testing data
train_data = pd.read_parquet('train.parquet')
test_data = pd.read_parquet('test.parquet')

# Defining Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Making it 128 x 128 because this is the data requirement.
    transforms.RandomHorizontalFlip(),  # Data augmentation for training
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing it using ImageNet stats
])

# This is a class to handle the byte data in the .parquet file
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extracting the image bytes
        image_bytes = self.dataframe.iloc[idx]['image']['bytes']

        # Loading the image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Applying the transformations
        if self.transform:
            image = self.transform(image)

        # Getting the label
        label = self.dataframe.iloc[idx]['label']
        return image, label

# Creating the datasets and DataLoaders
train_dataset = ImageDataset(train_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ImageDataset(test_data, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Loading the model (it's pretrained), I am using ResNet50
model = models.resnet50(pretrained=True)

# Modifying the final layer to match the number of classes
num_classes = len(train_data['label'].unique())
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# I am using my GPU because it's faster to run a model on it so don't worry about this
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training for 10 epochs, 10 here is perfect, more is too much, and less is more prone to loss, I tested different numbers
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Saving the model after training
torch.save(model.state_dict(), 'resnet_model.pth')
print("Training complete. Model saved as 'resnet_model.pth'.")

# Evaluating the model on the test dataset
model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

average_test_loss = test_loss / len(test_loader)
accuracy = 100 * correct / total

print(f"Test Loss: {average_test_loss}")
print(f"Test Accuracy: {accuracy}%")
