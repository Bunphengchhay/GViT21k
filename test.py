import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoConfig, AutoModelForImageClassification
from sklearn.metrics import classification_report
import csv
from PIL import Image
from utils import CustomDataset  # You may need to adjust the import based on your project structure

# Constants
TEST_DIR = "RAFDB_Kaggle/DATASET/test"
TEST_LABELS = "RAFDB_Kaggle/test_labels.csv"
BATCH_SIZE = 32
NUM_CLASSES = 7  # Number of classes in your dataset (adjust as needed)

emotion_labels = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral"
}

# Data Transforms for Testing
def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the fine-tuned model's configuration
    config_path = "checkpointvitEnhancev1/model/config.json"
    config = AutoConfig.from_pretrained(config_path)

    # Load the fine-tuned model using SafeTensors and the loaded configuration
    model_checkpoint_path = "checkpointvitEnhancev1/model/model.safetensors"
    model = AutoModelForImageClassification.from_pretrained(model_checkpoint_path, config=config)
    model.to(device)
    model.eval()

    # Create a DataLoader for the test dataset
    test_transforms = get_test_transforms()
    test_dataset = CustomDataset(TEST_DIR, TEST_LABELS, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # Adjust labels to match the training label range (subtract 1)
            labels = labels.to(device) - 1

            # Perform inference
            outputs = model(images).logits

            # Get the predicted class for each image
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Map the predicted labels to their corresponding emotions
    predicted_emotions = [emotion_labels[pred] for pred in all_predictions]

    # Calculate the classification report
    report = classification_report(all_labels, all_predictions, target_names=emotion_labels.values())

    # Print the classification report
    print(report)

if __name__ == "__main__":
    main()






# # test.py
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch.nn.functional as F
# import utils
# from sklearn.metrics import confusion_matrix, classification_report
# from transformers import AutoConfig, AutoModelForImageClassification
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from safetensors import safe_open
# TRAIN_DIR = "RAFDB_Kaggle/DATASET/train"
# TEST_DIR = "RAFDB_Kaggle/DATASET/test"
# TRAIN_LABELS = "RAFDB_Kaggle/train_labels.csv"
# TEST_LABELS = "RAFDB_Kaggle/test_labels.csv"
# BATCH_SIZE = 32
# from torchvision import transforms
# # import CustomDataset

# # Set device for PyTorch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # def load_model(model_config_path, model_state_path):
# #     # Load the model configuration
# #     config = AutoConfig.from_pretrained(model_config_path)
    
# #     # Create the model
# #     model = AutoModelForImageClassification.from_config(config)
    
# #     # Load the model state
# #     model_state = torch.load(model_state_path, map_location=device)
# #     model.load_state_dict(model_state)

# #     return model

# # Paths to your model config and state
# config_path = 'checkpointvitEnhance/model/config.json'  # Update with your actual config file path
# model_state_path = 'checkpointvitEnhance/latest_checkpoint.pth'  # Update with your actual model state file path

# # # Load your model
# # # model = load_model(config_path, model_state_path)  # Update num_classes as needed
# # model = AutoModelForImageClassification.from_pretrained('checkpointvitEnhance/model')
# # model =  utils.CustomDataset( TEST_DIR, TEST_LABELS, transform=utils.get_test_transforms())
# # model.load_state_dict(torch.load('checkpointvitEnhance/final_model.pth', map_location=device))
# # Assuming you have a function to create an instance of your model
# model, feature_extractor = utils.load_model(num_classes=7, device=device)

# # Load the pre-trained model's state dictionary
# model.load_state_dict(torch.load('checkpointvitEnhance/final_model.pth', map_location=device))


# model.to(device)
# model.eval()

# test_loader = utils.get_test_data_loader()

# # Function to test the model
# def test_model(data_loader):
#     model.to(device)
#     all_predictions = []
#     all_labels = []
    
#     with torch.no_grad():
#         for images, labels in tqdm(data_loader):
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)

#             # Extract logits from the model's output
#             logits = outputs.logits

#             # Convert logits to predicted class indices
#             _, predicted = torch.max(logits, 1)
            
#             # Collect all predictions and labels
#             all_predictions.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
            
    
#         # Emotion labels
#     emotion_labels = {
#         0: "Surprise",
#         1: "Fear",
#         2: "Disgust",
#         3: "Happiness",
#         4: "Sadness",
#         5: "Anger",
#         6: "Neutral"
#     }
#     # Convert the emotion labels to a list for classification report
#     class_names = [emotion_labels[i] for i in range(len(emotion_labels))]
#     # Calculate and print metrics
#     print('Classification Report:')
#     print(classification_report(all_labels, all_predictions, target_names=class_names))  # Update label names as required
#     print('Confusion Matrix:')
#     cm = confusion_matrix(all_labels, all_predictions)
#     sns.heatmap(cm, annot=True, fmt='d')
#     plt.show()

# test_model(test_loader)



# Print classification report

# try:
#     best_model = torch.load('checkpointvitEnhance/best_model.pth')
#     print(best_model)
# except Exception as e:
#     print(f"An error occurred while loading the model: {e}")

# # Load the best model
# best_model = torch.load('checkpointvitEnhance/best_model.pth')

# # Set the best model to evaluation mode
# best_model.eval()

# # Use the same test dataset and dataloader
# test_dataset = utils.CustomDataset(TEST_DIR, TEST_LABELS, transform=utils.get_test_transforms())
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Evaluation loop for the best model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# best_model.to(device)
# test_loss = 0
# correct = 0
# total = 0
# all_labels = []
# all_predictions = []
# with torch.no_grad():
#     for images, labels in tqdm(test_dataloader):
#         images, labels = images.to(device), labels.to(device)
#         outputs = best_model(images).logits
#         loss = F.cross_entropy(outputs, labels)

#         test_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         all_labels.extend(labels.cpu().numpy())
#         all_predictions.extend(predicted.cpu().numpy())

# # Calculate average loss and accuracy
# test_loss /= len(test_dataloader)
# test_accuracy = 100 * correct / total
# print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# # Calculate and print the confusion matrix
# conf_matrix = confusion_matrix(all_labels, all_predictions)
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

# # Example usage
# emotion_labels = {
#     0: "Surprise",
#     1: "Fear",
#     2: "Disgust",
#     3: "Happiness",
#     4: "Sadness",
#     5: "Anger",
#     6: "Neutral"
# }
# # Calculate and print precision, recall, and F1-score
# print(classification_report(all_labels, all_predictions))

# # Print classification report
# print(classification_report(all_labels, all_predictions, target_names=emotion_labels.values(), zero_division=0))


# # test.py
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch.nn.functional as F
# import utils
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from safetensors import safe_open
# # Constants
# TRAIN_DIR = "RAFDB_Kaggle/DATASET/train"
# TEST_DIR = "RAFDB_Kaggle/DATASET/test"
# TRAIN_LABELS = "RAFDB_Kaggle/train_labels.csv"
# TEST_LABELS = "RAFDB_Kaggle/test_labels.csv"
# BATCH_SIZE = 32
# # from safetensors.core import SafeTensor
# # model = SafeTensor.load('/content/drive/MyDrive/APViT/models/apvit_tiny_p16_384.safetensor')
# # Load model and test dataloader
# # model = utils.load_model()
# # Load the tensors from the .safetensor file
# tensor = utils.load_tensors_from_safetensor("checkpointvitEnhance/model/model.safetensors")
# # Load the model
# model = utils.load_model_with_tensors("checkpointvitEnhance/model/config.json", tensor)

# test_dataset = utils.CustomDataset(TEST_DIR, TEST_LABELS, transform=utils.get_test_transforms())
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Evaluation loop
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# test_loss = 0
# correct = 0
# total = 0
# all_labels = []
# all_predictions = []
# with torch.no_grad():
#     for images, labels in tqdm(test_dataloader):
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images).logits
#         loss = F.cross_entropy(outputs, labels)

#         test_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         all_labels.extend(labels.cpu().numpy())
#         all_predictions.extend(predicted.cpu().numpy())

# # Calculate average loss and accuracy
# test_loss /= len(test_dataloader)
# test_accuracy = 100 * correct / total
# print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# # Calculate and print the confusion matrix
# conf_matrix = confusion_matrix(all_labels, all_predictions)
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

# # Example usage
# emotion_labels = {
#     0: "Surprise",
#     1: "Fear",
#     2: "Disgust",
#     3: "Happiness",
#     4: "Sadness",
#     5: "Anger",
#     6: "Neutral"
# }
# # Calculate and print precision, recall, and F1-score
# print(classification_report(all_labels, all_predictions))

# # Print classification report
# print(classification_report(all_labels, all_predictions, target_names=emotion_labels.values(), zero_division=0))




# import torch
# from sklearn.metrics import accuracy_score, classification_report
# import os
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from utils import CustomDataset, load_model, get_test_transforms, get_data_loaders

# # Configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the model and feature extractor
# num_classes = 7  # Update this based on your dataset
# model, feature_extractor = load_model(num_classes, device)

# # Set the model to evaluation mode
# model.eval()
# TRAIN_DIR = "RAFDB_Kaggle/DATASET/train"
# TEST_DIR = "RAFDB_Kaggle/DATASET/test"
# TRAIN_LABELS = "RAFDB_Kaggle/train_labels.csv"
# TEST_LABELS = "RAFDB_Kaggle/test_labels.csv"
# BATCH_SIZE = 32
# # Get the test data loader
# test_loader = DataLoader(
#     CustomDataset(TEST_DIR, TEST_LABELS, transform=get_test_transforms()),
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )
# # Initialize lists to store predictions and ground truths
# all_predictions = []
# all_labels = []

# # Iterate through the test dataset and make predictions
# with torch.no_grad():
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(images).logits

#         # Get the predicted class for each image
#         predictions = torch.argmax(outputs, dim=1).cpu().numpy()

#         all_predictions.extend(predictions)
#         all_labels.extend(labels.cpu().numpy())

# # Calculate accuracy and print classification report
# accuracy = accuracy_score(all_labels, all_predictions)
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(classification_report(all_labels, all_predictions))





# # test.py
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch.nn.functional as F
# import utils

# # from safetensors.core import SafeTensor
# # model = SafeTensor.load('/content/drive/MyDrive/APViT/models/apvit_tiny_p16_384.safetensor')
# # Load model and test dataloader
# # model = utils.load_model()
# # Load the tensors from the .safetensor file
# tensor = utils.load_tensors_from_safetensor("checkpoint/model.safetensors")
# # Load the model
# model = utils.load_model_with_tensors("checkpoint/config.json", tensor)

# test_dataset = utils.RAFDBDataset('RAF-DB/basic', 'test', transform=utils.test_transforms)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Evaluation loop
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# test_loss = 0
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in tqdm(test_dataloader):
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images).logits
#         loss = F.cross_entropy(outputs, labels)

#         test_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# # Calculate average loss and accuracy
# test_loss /= len(test_dataloader)
# test_accuracy = 100 * correct / total
# print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# import torch
# import utils
# import numpy as np
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # Configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes = 7  # Update this based on your dataset

# # Load model and data loaders
# model, feature_extractor = utils.load_model(num_classes, device)
# _, test_loader = utils.get_data_loaders()

# # Load saved model weights
# model.load_state_dict(torch.load("checkpointvitEnhance/best_model.pth"))

# # Initialize lists to store ground truth and predicted labels
# true_labels = []
# predicted_labels = []

# # Evaluate the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = feature_extractor(images, return_tensors="pt").to(device)
#         labels = labels.to(device)
#         outputs = model(images).logits
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#         # Store true and predicted labels
#         true_labels.extend(labels.cpu().numpy())
#         predicted_labels.extend(predicted.cpu().numpy())

#     accuracy = 100 * correct / total
#     print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# # Calculate and display the confusion matrix
# conf_matrix = confusion_matrix(true_labels, predicted_labels)
# class_names = [str(i) for i in range(num_classes)]  # Update class names accordingly
# ConfusionMatrixDisplay(conf_matrix, display_labels=class_names).plot(cmap='viridis', values_format='d')

# import torch
# import utils

# # Configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes = 7  # Update this based on your dataset

# # Load model and data loaders
# model, feature_extractor = utils.load_model(num_classes, device)
# _, test_loader = utils.get_data_loaders()

# # Load saved model weights
# model.load_state_dict(torch.load("vit_model.pth"))

# # Evaluate the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = feature_extractor(images, return_tensors="pt").to(device)
#         labels = labels.to(device)
#         outputs = model(images).logits
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print(f'Accuracy of the model on the test images: {100 * correct / total}%')
