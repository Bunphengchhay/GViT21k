import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoConfig, AutoModelForImageClassification
from sklearn.metrics import classification_report
import csv
from PIL import Image
from utils import CustomDataset
from tqdm import tqdm  # Import tqdm for the progress bar
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
        for images, labels in tqdm(test_loader, desc="Testing Progress"):
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