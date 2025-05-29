import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pickle as pk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd # Added for pandas DataFrame operations

# Define the same device as in your training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Re-define the Model Architecture ---
# This MUST be identical to the EnhancedNet class in cnn_final.py
class EnhancedNet(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.3): # Ensure num_classes matches the saved model
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

# --- Load the Model ---
MODEL_PATH = 'best_model1.pth' 
loaded_model = EnhancedNet(num_classes=5, dropout_rate=0.3) 
loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
loaded_model.to(device)
loaded_model.eval()
print("Model loaded successfully and set to evaluation mode.")

# --- Data Loading and Preparation for Evaluation ---
# Custom dataset for labeled data (spectrogram, label, filename)
class AccentDataset(Dataset):
    def __init__(self, data):
        # Expects data to be a list of (spectrogram, label, filename_string)
        self.data = data  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming data items are (spectrogram, label, filename)
        if len(self.data[idx]) < 3:
            # Fallback or error if filename is missing
            # This part depends on how your data is structured if filenames are not always present
            print(f"Warning: Item at index {idx} does not contain filename. Gender analysis might be incomplete.")
            spectro, label = self.data[idx] # Original behavior
            filename = "" # Placeholder
        else:
            spectro, label, filename = self.data[idx]
        
        tensor = torch.from_numpy(spectro).float().unsqueeze(0)
        processed_label = int(label) - 1 # Convert labels from range 1-5 to range 0-4
        
        gender = 'unknown' # Default gender
        if filename and len(filename) > 1:
            # Assuming filename format like "1f_..." or "2m_..." where gender is the second character
            char_gender = filename[1].lower()
            if char_gender == 'f':
                gender = 'female'
            elif char_gender == 'm':
                gender = 'male'
        
        return tensor, processed_label, gender


# Collate function for labeled data (now handles gender)
def pad_collate_labeled(batch):
    # Batch items are (tensor, processed_label, gender)
    inputs, labels, genders = zip(*batch)
    
    max_t = max(t.size(2) for t in inputs)
    padded_inputs = [F.pad(t, (0, max_t - t.size(2), 0, 0)) for t in inputs]
    inputs_tensor = torch.stack(padded_inputs)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return inputs_tensor, labels_tensor, list(genders)


# Load training data (which includes samples for validation)
print('Loading full training data for evaluation split...')
try:
    with open('spectro_data_train_7979_filesls.pkl', 'rb') as f:
        full_train_data = pk.load(f)
    # Verify the structure of the first item to ensure it contains filenames
    if full_train_data and len(full_train_data[0]) < 3:
        print("Warning: Loaded data items do not seem to contain filenames. Gender analysis will likely fail or be inaccurate.")
        print("Expected data structure: (spectrogram_array, accent_label, filename_string)")
        print(f"Actual structure of first item: {type(full_train_data[0])} with {len(full_train_data[0])} elements")

except FileNotFoundError:
    print("Error: 'spectro_data_train_7979_filesls.pkl' not found. Please ensure the training data file is in the correct path.")
    exit()
except Exception as e:
    print(f"Error loading or verifying 'spectro_data_train_7979_filesls.pkl': {e}")
    exit()


# Split training data into train and validation sets (consistent with cnn_final.py)
val_share = 0.2 
val_size = int(val_share * len(full_train_data))
train_size = len(full_train_data) - val_size

# Ensure a fixed split for consistent evaluation
generator = torch.Generator().manual_seed(0) # Use a fixed seed for reproducibility
train_subset_data, val_subset_data = random_split(full_train_data, [train_size, val_size], generator=generator)

# Create DataLoaders for evaluation
eval_train_dataset = AccentDataset(train_subset_data)
eval_val_dataset = AccentDataset(val_subset_data)

eval_train_loader = DataLoader(eval_train_dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=pad_collate_labeled)
eval_val_loader = DataLoader(eval_val_dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=pad_collate_labeled)


# --- Performance Evaluation Function ---
def evaluate_model_performance(model, dataloader, dataset_name):
    print(f"\n--- Evaluating Performance on {dataset_name} ---")
    model.eval()
    all_labels = []
    all_predictions = []
    all_genders = []

    with torch.no_grad():
        for inputs, labels, batch_genders in dataloader: # Expecting gender from collate_fn
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_genders.extend(batch_genders)


    if not all_labels:
        print(f"No data found in {dataset_name} to evaluate.")
        return [], [], [] # Return empty lists

    accuracy = accuracy_score(all_labels, all_predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)

    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Precision (Macro): {precision_macro:.4f}")
    print(f"Overall Recall (Macro): {recall_macro:.4f}")
    print(f"Overall F1-score (Macro): {f1_macro:.4f}")
    print(f"Overall Precision (Weighted): {precision_weighted:.4f}")
    print(f"Overall Recall (Weighted): {recall_weighted:.4f}")
    print(f"Overall F1-score (Weighted): {f1_weighted:.4f}")

    print("\nPer-Class Performance (Accent-wise):")
    unique_labels_for_report = sorted(list(set(all_labels) | set(all_predictions))) 
    class_names_display = [f"Accent {i+1}" for i in unique_labels_for_report]
    
    report = classification_report(all_labels, all_predictions, labels=unique_labels_for_report, target_names=class_names_display, zero_division=0)
    print(report)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels_for_report)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names_display, yticklabels=class_names_display)
    plt.title(f'{dataset_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    filename_safe_dataset_name = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plt.savefig(f'{filename_safe_dataset_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close() 
    
    return all_labels, all_predictions, all_genders


# --- Run Evaluations ---
# We don't need to store results for training set for gender analysis, but function returns 3 values now
_, _, _ = evaluate_model_performance(loaded_model, eval_train_loader, "Training Set (Subset)")
val_labels, val_predictions, val_genders = evaluate_model_performance(loaded_model, eval_val_loader, "Validation Set")

# --- Gender-based Performance Analysis for Validation Set ---
if val_labels and val_genders: # Check if we have data
    print("\n--- Analyzing Performance by Gender on Validation Set ---")
    
    val_analysis_df = pd.DataFrame({
        'true_label': val_labels,      # 0-4 range
        'predicted_label': val_predictions, # 0-4 range
        'gender': val_genders
    })
    val_analysis_df['correct_prediction'] = (val_analysis_df['true_label'] == val_analysis_df['predicted_label'])
    val_analysis_df['true_accent'] = val_analysis_df['true_label'] + 1 # Map to 1-5 for reporting

    # Gender-specific accuracy
    gender_accuracy_summary = val_analysis_df.groupby('gender')['correct_prediction'].agg(
        Accuracy='mean',
        Sample_Count='size'
    ).reset_index()

    print("\nAccuracy by Gender (Validation Set):")
    print(gender_accuracy_summary)

    # Plotting gender accuracy
    if not gender_accuracy_summary.empty:
        fig_gender, ax_gender = plt.subplots(figsize=(7, 5))
        
        # Determine colors, ensuring 'male' and 'female' get consistent colors if present
        gender_order = [g for g in ['male', 'female', 'unknown'] if g in gender_accuracy_summary['gender'].values]
        plot_df = gender_accuracy_summary.set_index('gender').reindex(gender_order).reset_index()

        colors = []
        for gender_val in plot_df['gender']:
            if gender_val == 'male':
                colors.append('blue')
            elif gender_val == 'female':
                colors.append('red')
            else:
                colors.append('grey')

        plot_df.plot(x='gender', y='Accuracy', kind='bar', ax=ax_gender, color=colors, legend=False)
        
        ax_gender.set_title('Model Accuracy by Gender (Validation Set)')
        ax_gender.set_xlabel('Gender')
        ax_gender.set_ylabel('Accuracy')
        ax_gender.set_ylim(0, 1)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('validation_accuracy_by_gender.png', dpi=300)
        print("Gender accuracy plot saved to validation_accuracy_by_gender.png")
        plt.close(fig_gender)
    else:
        print("No gender data to plot.")

    # Combined Accent-Gender Performance
    if 'unknown' in val_analysis_df['gender'].unique():
        print("\nNote: 'unknown' gender category present in combined performance.")
        
    combined_performance = val_analysis_df.groupby(['true_accent', 'gender']).agg(
        Accuracy=('correct_prediction', 'mean'),
        Sample_Count=('correct_prediction', 'size') # Count of samples in each group
    ).round(3)
    
    print("\nCombined Accent-Gender Performance (Validation Set):")
    print(combined_performance)

else:
    print("\nNot enough data from validation set to perform gender analysis.")


# --- Inference on Unlabeled Test Set ---
class SpectrogramDataset(Dataset):
    def __init__(self, data):
        self.data = data # Expects list of spectrogram numpy arrays
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        spectro = self.data[idx]
        tensor = torch.from_numpy(spectro).float().unsqueeze(0)
        return tensor

def pad_collate_test(batch): # Batch is a list of tensors
    inputs = batch 
    max_t = max(t.size(2) for t in inputs)
    padded = [F.pad(t, (0, max_t - t.size(2), 0, 0)) for t in inputs]
    inputs_tensor = torch.stack(padded)
    return inputs_tensor

print('\n--- Generating predictions for the unlabeled test set ---')
try:
    with open('spectro_data_test.pkl', 'rb') as f:
        test_data_for_inference = pk.load(f) # This should be a list of spectrograms

    testset_for_inference = SpectrogramDataset(test_data_for_inference)
    testloader_for_inference = DataLoader(testset_for_inference, batch_size=16, shuffle=False, num_workers=2, collate_fn=pad_collate_test)

    all_final_predictions = []
    with torch.no_grad():
        for inputs_batch in testloader_for_inference: 
            inputs_batch = inputs_batch.to(device)
            outputs = loaded_model(inputs_batch)
            _, preds = torch.max(outputs, 1)
            all_final_predictions.extend(preds.cpu().tolist()) # Predictions are 0-4

    print(f'Generated {len(all_final_predictions)} predictions for the unlabeled test set.')
    
    csv_file_path = 'test_set_predictions.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["SampleID", "PredictedClass"])
        for i, pred in enumerate(all_final_predictions):
            csv_writer.writerow([i, pred + 1]) # Convert 0-4 to 1-5 for output
            
    print(f'Unlabeled test set predictions saved to {csv_file_path}')

except FileNotFoundError:
    print("Error: 'spectro_data_test.pkl' not found. Skipping inference on unlabeled test set.")
except Exception as e:
    print(f"An error occurred during inference on unlabeled test set: {e}")

# filepath: c:\Users\zboom\Documents\Uni\Year 3\Deep Learning\DeepLearning copy\DeepLearning copy\code\test_script_final.p