import torch
from torchvision import transforms
from transformers import AutoConfig, AutoModelForImageClassification
from PIL import Image

# Constants
MODEL_DIR = "checkpointvitEnhancev1/model"  # Path to your model directory
BATCH_SIZE = 1  # Set batch size to 1 for single image inference
NUM_CLASSES = 7  # Number of classes in your dataset (adjust as needed)

emotion_labels = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral"
}

# Data Transforms for Inference
def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# def predict_emotion(image_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load the fine-tuned model's configuration
#     config_path = f"{MODEL_DIR}/config.json"
#     config = AutoConfig.from_pretrained(config_path)

#     # Load the fine-tuned model using SafeTensors and the loaded configuration
#     model_checkpoint_path = f"{MODEL_DIR}/model.safetensors"
#     model = AutoModelForImageClassification.from_pretrained(model_checkpoint_path, config=config)
#     model.to(device)
#     model.eval()

#     # Load and preprocess the input image
#     inference_transforms = get_inference_transforms()
#     image = Image.open(image_path).convert("RGB")
#     image = inference_transforms(image).unsqueeze(0).to(device)

#     # Perform inference
#     with torch.no_grad():
#         outputs = model(image).logits

#     # Get the predicted class for the image
#     _, predicted = torch.max(outputs, 1)
    
#     # Map the predicted label to its corresponding emotion
#     predicted_emotion = emotion_labels[predicted.item()]

#     return predicted_emotion

def predict_emotion(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the fine-tuned model's configuration
    config_path = f"{MODEL_DIR}/config.json"
    config = AutoConfig.from_pretrained(config_path)

    # Load the fine-tuned model using SafeTensors and the loaded configuration
    model_checkpoint_path = f"{MODEL_DIR}/model.safetensors"
    model = AutoModelForImageClassification.from_pretrained(model_checkpoint_path, config=config)
    model.to(device)
    model.eval()

    # Load and preprocess the input image
    inference_transforms = get_inference_transforms()
    image = Image.open(image_path).convert("RGB")
    image = inference_transforms(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image).logits

    # Get the predicted class for the image and adjust by adding 1
    _, predicted = torch.max(outputs, 1)
    predicted_label = predicted.item() + 1
    
    # Map the predicted label to its corresponding emotion
    predicted_emotion = emotion_labels[predicted_label]

    return predicted_emotion


if __name__ == "__main__":
    # Provide the path to the input image you want to predict
    # input_image_path = "RAFDB_Kaggle/DATASET/test/4/test_2224_aligned.jpg"
    input_image_path = "RAFDB_Kaggle/DATASET/test/6/test_0228_aligned.jpg"
    
    predicted_emotion = predict_emotion(input_image_path)
    print(f"Predicted Emotion: {predicted_emotion}")


# import torch
# import argparse
# from torchvision import transforms
# from PIL import Image
# from transformers import AutoConfig, AutoModelForImageClassification
# from utils import CustomDataset  # You may need to adjust the import based on your project structure

# # Constants
# MODEL_DIR = "checkpointvitEnhancev1/model"  # Path to the model directory
# EMOTION_LABELS = {
#     0: "Surprise",
#     1: "Fear",
#     2: "Disgust",
#     3: "Happiness",
#     4: "Sadness",
#     5: "Anger",
#     6: "Neutral"
# }

# def load_model(model_dir):
#     # Load the fine-tuned model's configuration
#     config_path = f"{model_dir}/config.json"
#     config = AutoConfig.from_pretrained(config_path)

#     # Load the fine-tuned model using SafeTensors and the loaded configuration
#     model_checkpoint_path = f"{model_dir}/model.safetensors"
#     model = AutoModelForImageClassification.from_pretrained(model_checkpoint_path, config=config)

#     return model

# def preprocess_image(image_path):
#     # Define data transforms for the input image
#     image_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     # Load and preprocess the image
#     image = Image.open(image_path).convert("RGB")
#     image = image_transforms(image).unsqueeze(0)  # Add batch dimension

#     return image

# def predict_emotion(image_path, model):
#     # Load and preprocess the input image
#     image = preprocess_image(image_path)

#     # Perform inference
#     with torch.no_grad():
#         outputs = model(image).logits
#         _, predicted = torch.max(outputs, 1)
#         predicted_label = EMOTION_LABELS[predicted.item()]

#     return predicted_label

# def main():
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description="Emotion Prediction from Images")
#     parser.add_argument("image_path", type=str, help="Path to the input image")
#     args = parser.parse_args()

#     # Load the pre-trained model
#     model = load_model(MODEL_DIR)

#     # Predict emotion for the input image
#     predicted_label = predict_emotion(args.image_path, model)

#     print(f"Predicted Emotion: {predicted_label}")

# if __name__ == "__main__":
#     main()
