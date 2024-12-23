import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from cuml.linear_model import LogisticRegression
from cuml.preprocessing import StandardScaler
import numpy as np
import cupy as cp
from numba import cuda
import time
import nvtx


@cuda.jit
def preprocess_target_kernel(target, processed_target):
    idx = cuda.grid(1)
    if idx < target.size:
        val = target[idx]
        if val == 4:
            processed_target[idx] = 1
        else:
            processed_target[idx] = 0

@nvtx.annotate("preprocess_target", color="red")
def preprocess_target_parallel(target_column):
    start_time = time.time()

    # Convert target column to a NumPy array
    target_array = target_column.to_numpy(dtype=np.int32)

    # Allocate GPU memory for the output
    processed_target = np.zeros_like(target_array, dtype=np.int32)
    d_target = cuda.to_device(target_array)
    d_processed_target = cuda.to_device(processed_target)

    # Launch kernel
    threads_per_block = 256
    blocks_per_grid = (len(target_array) + threads_per_block - 1) // threads_per_block
    preprocess_target_kernel[blocks_per_grid, threads_per_block](d_target, d_processed_target)

    # Copy back to host
    processed_target = d_processed_target.copy_to_host()
    end_time = time.time()
    print(f"preprocess_target_parallel executed in {end_time - start_time:.4f} seconds")
    return processed_target


@cuda.jit
def tokenize_kernel(input_texts, feature_matrix, max_tokens, max_text_length):
    row = cuda.grid(1)
    if row < input_texts.shape[0]:
        text = input_texts[row]
        token_count = 0
        current_token_length = 0
        for i in range(len(text)):
            if text[i] == b' ' or i == len(text) - 1:  # Token boundary
                if token_count < max_tokens:
                    feature_matrix[row, token_count] = current_token_length
                    token_count += 1
                    current_token_length = 0
            else:
                current_token_length += 1

@nvtx.annotate("tokenize_texts", color="purple")
def tokenize_texts_parallel(input_texts, max_tokens=100, max_text_length=256):
    start_time = time.time()

    # Prepare input text as fixed-length byte arrays
    text_array = np.full((len(input_texts), max_text_length), b' ', dtype='S1')
    for i, text in enumerate(input_texts):
        encoded_text = text.encode('utf-8')[:max_text_length]
        text_array[i, :len(encoded_text)] = np.frombuffer(encoded_text, dtype='S1')

    # Initialize GPU memory
    feature_matrix = np.zeros((len(input_texts), max_tokens), dtype=np.float32)
    d_text_array = cuda.to_device(text_array)
    d_feature_matrix = cuda.to_device(feature_matrix)

    # Define CUDA grid dimensions
    threads_per_block = 256
    blocks_per_grid = (len(input_texts) + threads_per_block - 1) // threads_per_block

    # Launch the kernel
    tokenize_kernel[blocks_per_grid, threads_per_block](d_text_array, d_feature_matrix, max_tokens, max_text_length)

    # Copy back to host
    feature_matrix = d_feature_matrix.copy_to_host()
    end_time = time.time()
    print(f"tokenize_texts_parallel executed in {end_time - start_time:.4f} seconds")
    return feature_matrix


def split_data_parallel(X, y, test_size=0.2, random_state=42):
    start_time = time.time()
    np.random.seed(random_state)

    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Split the data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    end_time = time.time()
    print(f"split_data_parallel executed in {end_time - start_time:.4f} seconds")
    return X_train, X_test, y_train, y_test

@nvtx.annotate("train_model", color="cyan")
def train_model_gpu(X_train_features, y_train):
    start_time = time.time()

    # Convert data to GPU arrays
    X_train_features = cp.array(X_train_features)
    y_train = cp.array(y_train)

    # Scale features on GPU
    scaler = StandardScaler()
    X_train_features = scaler.fit_transform(X_train_features)

    # Train logistic regression on GPU
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_features, y_train)

    end_time = time.time()
    print(f"train_model_gpu executed in {end_time - start_time:.4f} seconds")
    return model, scaler


def evaluate_model_gpu(model, scaler, X_test_features, y_test):
    start_time = time.time()

    # Convert test data to GPU arrays
    X_test_features = cp.array(X_test_features)

    # Scale features on GPU
    X_test_features = scaler.transform(X_test_features)

    # Predict on GPU
    y_pred = model.predict(X_test_features)
    y_pred = y_pred.get()  # Convert back to NumPy for evaluation

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}")
    end_time = time.time()
    print(f"evaluate_model_gpu executed in {end_time - start_time:.4f} seconds")


def main():
    total_start_time = time.time()

    print("Loading dataset...")
    DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv('sampled_dataset.csv', names=DATASET_COLUMNS, header=0, encoding="ISO-8859-1")
    df = df[['text', 'target']].dropna().drop_duplicates()

    print("Preprocessing target column on GPU...")
    y = preprocess_target_parallel(df['target'])

    print("Tokenizing texts on GPU...")
    X = tokenize_texts_parallel(df['text'])

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data_parallel(X, y)

    print("Training model on GPU...")
    model, scaler = train_model_gpu(X_train, y_train)

    print("Evaluating model on GPU...")
    evaluate_model_gpu(model, scaler, X_test, y_test)

    total_end_time = time.time()
    print(f"Total execution time: {total_end_time - total_start_time:.4f} seconds")


if __name__ == "__main__":
    main()
