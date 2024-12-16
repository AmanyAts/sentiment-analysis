import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import nvtx  # Import NVTX for profiling
from numba import cuda  # Import Numba for CUDA
import psutil
from torch.cuda.amp import GradScaler, autocast

@cuda.jit
def preprocess_target_chunk(target, output):
    """CUDA kernel to preprocess target values in a chunk."""
    idx = cuda.grid(1)
    if idx < target.size:
        output[idx] = 1 if target[idx] == 4 else target[idx]

def download_and_load_dataset():
    """Loads and preprocesses the Sentiment140 dataset in chunks."""
    with nvtx.annotate("Function: download_and_load_dataset", color="blue"):
        DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
        DATASET_ENCODING = "ISO-8859-1"
        chunk_size = 100000
        texts, targets = [], []

        for chunk in pd.read_csv('sampled_dataset.csv', encoding=DATASET_ENCODING, header=0, names=DATASET_COLUMNS, chunksize=chunk_size):
            chunk = chunk[['text', 'target']].dropna().drop_duplicates()

            # Prepare target preprocessing
            target_array = chunk['target'].values
            output_array = cuda.to_device(np.zeros_like(target_array))
            d_target = cuda.to_device(target_array)

            threads_per_block = 256
            blocks_per_grid = (target_array.size + threads_per_block - 1) // threads_per_block
            preprocess_target_chunk[blocks_per_grid, threads_per_block](d_target, output_array)

            # Copy processed target back to host
            chunk['target'] = output_array.copy_to_host()

            texts.extend(chunk['text'].tolist())
            targets.extend(chunk['target'].tolist())

        return texts, targets

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets with class balancing."""
    with nvtx.annotate("Function: split_data", color="green"):
        class_counts = Counter(y)
        valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
        X_filtered = [x for x, label in zip(X, y) if label in valid_classes]
        y_filtered = [label for label in y if label in valid_classes]

        return train_test_split(X_filtered, y_filtered, test_size=test_size, random_state=random_state, stratify=y_filtered)

def tokenize_data(X_train, X_test):
    """Tokenizes training and testing data using DistilBERT tokenizer."""
    with nvtx.annotate("Function: tokenize_data", color="yellow"):
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128, return_tensors="pt")
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128, return_tensors="pt")
        return train_encodings, test_encodings

def prepare_data_for_gpu(train_encodings, test_encodings, y_train, y_test):
    """Prepares datasets for PyTorch DataLoader with GPU compatibility."""
    with nvtx.annotate("Function: prepare_data_for_gpu", color="orange"):
        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train))
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(y_test))

        # DataLoader automatically batches data
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        return train_loader, test_loader

def train_model(train_loader, test_loader, learning_rate=5e-5, epochs=3):
    """Trains a DistilBERT model for sentiment analysis on GPUs."""
    with nvtx.annotate("Function: train_model", color="red"):
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        model = nn.DataParallel(model)
        model = model.to("cuda")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler()

        for epoch in range(epochs):
            with nvtx.annotate(f"Epoch {epoch+1}", color="yellow"):
                model.train()
                for batch_idx, batch in enumerate(train_loader):
                    with nvtx.annotate(f"Batch {batch_idx}", color="blue"):
                        input_ids, attention_mask, labels = [x.to("cuda") for x in batch]
                        optimizer.zero_grad()
                        with autocast():
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            loss = loss_fn(outputs.logits, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

        torch.cuda.empty_cache()

def evaluate_model(model, test_loader):
    """Evaluates the model on the test dataset."""
    with nvtx.annotate("Function: evaluate_model", color="purple"):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = [x.to("cuda") for x in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        print(f"Test Accuracy: {correct / total:.2%}")

def main():
    with nvtx.annotate("Function: main", color="cyan"):
        print("Loading and preprocessing dataset...")
        X, y = download_and_load_dataset()  # Chunked processing

        print("Dataset Size: Rows =", len(X))
        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = split_data(X, y)

        print("Tokenizing data...")
        train_encodings, test_encodings = tokenize_data(X_train, X_test)

        print("Preparing DataLoader...")
        train_loader, test_loader = prepare_data_for_gpu(train_encodings, test_encodings, y_train, y_test)

        print("Training model...")
        train_model(train_loader, test_loader)

        print("Evaluating model...")
        evaluate_model(train_model, test_loader)

        print(f"Memory Usage: {psutil.virtual_memory().percent}%")

if __name__ == "__main__":
    main()
