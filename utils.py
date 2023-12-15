import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification, AutoConfig, AutoModelForImageClassification
from PIL import Image
import csv
from safetensors import safe_open

# Constants
TRAIN_DIR = "RAFDB_Kaggle/DATASET/train"
TEST_DIR = "RAFDB_Kaggle/DATASET/test"
TRAIN_LABELS = "RAFDB_Kaggle/train_labels.csv"
TEST_LABELS = "RAFDB_Kaggle/test_labels.csv"
BATCH_SIZE = 32

class CustomDataset(Dataset):
    def __init__(self, images_folder, labels_file, transform=None):
        self.images_folder = images_folder
        self.transform = transform

        # Load labels
        self.image_labels = []
        with open(labels_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)  # Skip the header row if there is one
            for row in csv_reader:
                img_name, label = row
                img_name = str(img_name)  # Convert img_name to string
                self.image_labels.append((img_name, int(label)))

    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        img_path = os.path.join(self.images_folder, str(label), img_name)  # Convert label to string
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Custom Dataset
# class CustomDataset(Dataset):
#     def __init__(self, images_folder, labels_file, transform=None):
#         self.images_folder = images_folder
#         self.transform = transform

#         # Load labels
#         self.image_labels = []
#         with open(labels_file, 'r') as file:
#             csv_reader = csv.reader(file)
#             next(csv_reader, None)  # Skip the header row if there is one
#             for row in csv_reader:
#                 img_name, label = row
#                 self.image_labels.append((img_name, int(label)))

#     def __len__(self):
#         return len(self.image_labels)
    
#     def __getitem__(self, idx):
#         img_name, label = self.image_labels[idx]
#         img_path = os.path.join(self.images_folder, str(label), img_name)
#         image = Image.open(img_path).convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         if label - 1 > 0:
#             label = label - 1

#         return image, label


    # def __getitem__(self, idx):
    #     img_name, label = self.image_labels[idx]
    #     img_path = os.path.join(self.images_folder, str(label), img_name)
    #     image = Image.open(img_path).convert("RGB")

    #     if self.transform:
    #         image = self.transform(image)

    #     if label -1 > 0:
    #         label = label -1

    #     return image, label

# Model Loading
def load_model(num_classes, device):
    model_path = "google/pretrainmodel"
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/feature_extractor")
    model = ViTForImageClassification.from_pretrained(
        model_path, 
        num_labels=num_classes, 
        ignore_mismatched_sizes=True  # Add this argument
    )
    model.to(device)
    return model, feature_extractor

# Data Transforms
def get_transforms():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Data Transforms with Augmentation for Training
def get_train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# # Data Transforms for Testing (without Augmentation)
# def get_test_transforms():
#     return transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_test_data_loader():
    test_transforms = get_test_transforms()
    test_dataset = CustomDataset(TEST_DIR, TEST_LABELS, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader



# Data Loaders
def get_data_loaders():
    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    train_dataset = CustomDataset(TRAIN_DIR, TRAIN_LABELS, transform=train_transforms)
    test_dataset = CustomDataset(TEST_DIR, TEST_LABELS, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

def load_tensors_from_safetensor(file_path, device=0):
    tensors = {}
    with safe_open(file_path, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

def load_model_with_tensors(model_config_path, tensors, num_classes=7):
    # Load the model configuration
    config = AutoConfig.from_pretrained(model_config_path, num_labels=num_classes)
    
    # Initialize the model with the loaded configuration
    model = AutoModelForImageClassification.from_config(config)

    # Create a new state_dict with the loaded tensors
    new_state_dict = {}
    for k, v in tensors.items():
        new_state_dict[k] = v

    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict)

    return model


def load_model_with_state_dict(model_config_path, state_dict_path, num_classes=7):
    # Load the model configuration
    config = AutoConfig.from_pretrained(model_config_path, num_labels=num_classes)
    
    # Initialize the model with the loaded configuration
    model = AutoModelForImageClassification.from_config(config)

    # Load the state_dict from the specified path
    state_dict = torch.load(state_dict_path)

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    return model
