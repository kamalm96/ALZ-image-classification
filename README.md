# ResNet50 Image Classification
This repository contains the code and data necessary to train and evaluate an image classification model using a pre-trained ResNet50 architecture. The model is fine-tuned to classify images stored in Parquet files, which contain image data in byte format. The project includes data loading, preprocessing, model training, evaluation, and saving of the trained model.

##Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Model Saving](#model-saving)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
## Installation
To run this project, you need to have Python installed along with the necessary libraries. You can install the required libraries using the following command:

bash
Copy code
pip install pandas pillow torchvision torch
Ensure that you have a GPU available if you want to speed up the training process.

## Data Preparation
The training and testing datasets are stored in Parquet files (train.parquet and test.parquet). Each dataset contains image data in byte format and corresponding labels.

To load and prepare the data:

python
Copy code
import pandas as pd

train_data = pd.read_parquet('train.parquet')
test_data = pd.read_parquet('test.parquet')
The data is then transformed using torchvision.transforms to resize, augment, and normalize the images according to ImageNet statistics.

## Model Training
The model is built using a pre-trained ResNet50 architecture. The final fully connected layer is modified to match the number of classes in the dataset.

Training is performed for 10 epochs, with the following key steps:

Images and labels are loaded in batches using DataLoader.
The model is trained using the Adam optimizer and CrossEntropyLoss.
The model's performance (loss) is tracked and printed after each epoch.
To train the model:


### Training loop
```
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
```
## Evaluation
After training, the model is evaluated on the test dataset to measure its performance.

To evaluate the model:
```
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
```
## Model Saving
The trained model is saved to disk for later use:
```
torch.save(model.state_dict(), 'resnet_model.pth')
print("Training complete. Model saved as 'resnet_model.pth'.")
```
## Results
After 10 epochs of training, the model's performance on the test dataset is printed, including the loss and accuracy.

Example output:
Epoch [x->10/10], Loss: y
Test Loss: y
Test Accuracy: z%
These metrics can vary depending on the specific dataset and hyperparameters used. The dataset used is included in this repo.
@dataset{alzheimer_mri_dataset,
  author = {Falah.G.Salieh},
  title = {Alzheimer MRI Dataset},
  year = {2023},
  publisher = {Hugging Face},
  version = {1.0},
  url = {https://huggingface.co/datasets/Falah/Alzheimer_MRI}
}


## Acknowledgments
This project uses the ResNet50 architecture provided by the PyTorch torchvision.models module, pre-trained on ImageNet. The data loading and transformation processes utilize Pandas and PIL libraries.
