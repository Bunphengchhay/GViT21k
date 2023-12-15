import torch
import torch.nn as nn
import torch.optim as optim
import os
import utils
from tqdm import tqdm

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7  # Update this based on your dataset
num_epochs = 25  # Adjust as needed
checkpoint_dir = "checkpointvitEnhancev1"
os.makedirs(checkpoint_dir, exist_ok=True)
best_val_loss = float('inf')
patience, trials = 5, 0  # Early stopping settings

# Load model and data loaders
model, feature_extractor = utils.load_model(num_classes, device)
train_loader, val_loader = utils.get_data_loaders()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

# Check for existing checkpoint
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
if os.path.isfile(latest_checkpoint_path):
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0
    print("No checkpoint found. Starting training from scratch.")
model.to(device)
# Training Loop
model.train()
for epoch in range(start_epoch, num_epochs):
    total_loss = 0
    total_accuracy = 0

    # Wrap train_loader with tqdm to create a progress bar
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)

    for images, labels in progress_bar:
        images = images.to(device)  # Images are already transformed and normalized
        labels = labels.to(device) - 1  # Adjust labels

        # Forward pass
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        accuracy = calculate_accuracy(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy

        # Update the progress bar
        progress_bar.set_postfix(loss=loss.item(), acc=accuracy)

    progress_bar.close()

    # Step the scheduler
    scheduler.step()

    # Average loss and accuracy
    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = total_accuracy / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device) - 1  # Adjust labels
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            accuracy = calculate_accuracy(outputs, labels)
            val_loss += loss.item()
            val_accuracy += accuracy

    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.2f}%")

    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        trials = 0
    else:
        trials += 1
        if trials >= patience:
            print(f"Early stopping triggered. Stopping training at epoch {epoch+1}")
            break

    # Save latest model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, latest_checkpoint_path)

# Save final model state
final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")
print("Training complete.")
# Save the final model
model.save_pretrained("checkpointvitEnhancev1/model")



# # train.py
# import os
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch.nn.functional as F
# import utils

# # Initialize model, optimizer, scheduler, and dataloaders
# model = utils.load_model()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# train_dataset = utils.RAFDBDataset('path_to_your_dataset', 'train', transform=utils.train_transforms)
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # Setup device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Load the checkpoint if it exists
# checkpoint_dir = "checkpoint"
# os.makedirs(checkpoint_dir, exist_ok=True)
# latest_checkpoint = utils.find_latest_checkpoint(checkpoint_dir)
# start_epoch = utils.load_checkpoint(latest_checkpoint, model, optimizer, device) if latest_checkpoint else 0

# # Training loop
# for epoch in range(start_epoch, 50):  # Adjust number of epochs as needed
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0

#     for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = F.cross_entropy(outputs, labels)  # Assuming your model outputs raw scores
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         correct += (predicted == labels).sum().item()
#         total += labels.size(0)

#     average_loss = total_loss / len(train_dataloader)
#     accuracy = 100 * correct / total
#     print(f'Epoch: {epoch+1}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

#     utils.save_checkpoint(model, optimizer, epoch, average_loss, accuracy, checkpoint_dir)

#     scheduler.step()

# print("Training complete.")
