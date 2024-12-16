import cudf
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numba import cuda
import numpy as np
import nvtx
import cupy as cp


# CUDA Kernel for Tokenization
@cuda.jit
def tokenize_kernel(input_texts, feature_matrix, max_tokens):
    row = cuda.grid(1)
    if row < input_texts.shape[0]:
        text = input_texts[row]
        token_count = 0
        start_idx = -1

        for i in range(input_texts.shape[1]):
            if text[i] != b' ' and start_idx == -1:  # Start of a token
                start_idx = i
            elif (text[i] == b' ' or i == input_texts.shape[1] - 1) and start_idx != -1:
                if token_count < max_tokens:
                    feature_matrix[row, token_count] = start_idx
                    token_count += 1
                start_idx = -1  # Reset for the next token


# CUDA Kernel for Target Preprocessing
@cuda.jit
def preprocess_target_kernel(target, processed_target):
    idx = cuda.grid(1)
    if idx < target.size:
        processed_target[idx] = 1 if target[idx] == 4 else 0


@nvtx.annotate("preprocess_target", color="red")
def preprocess_target(df):
    df = df[df['target'].str.isnumeric()]
    df['target'] = df['target'].astype('int32')

    target_array = df['target'].to_numpy()
    processed_target = np.zeros_like(target_array, dtype=np.int32)

    d_target = cuda.to_device(target_array)
    d_processed_target = cuda.to_device(processed_target)

    threads_per_block = 256
    blocks_per_grid = (target_array.size + threads_per_block - 1) // threads_per_block
    preprocess_target_kernel[blocks_per_grid, threads_per_block](d_target, d_processed_target)

    df['target'] = cudf.Series(d_processed_target.copy_to_host())
    return df


@nvtx.annotate("tokenize_texts", color="purple")
def tokenize_texts_gpu(input_texts, max_tokens=100, max_text_length=256):
    text_array = np.full((len(input_texts), max_text_length), b' ', dtype='S1')

    for i, text in enumerate(input_texts.to_pandas()):
        truncated_text = text.encode('utf-8')[:max_text_length]
        text_array[i, :len(truncated_text)] = np.frombuffer(truncated_text, dtype='S1')

    feature_matrix = cp.zeros((len(input_texts), max_tokens), dtype=cp.float32)

    d_text_array = cuda.to_device(text_array)
    d_feature_matrix = cuda.to_device(feature_matrix)

    threads_per_block = 256
    blocks_per_grid = (len(input_texts) + threads_per_block - 1) // threads_per_block
    tokenize_kernel[blocks_per_grid, threads_per_block](d_text_array, d_feature_matrix, max_tokens)

    feature_matrix = d_feature_matrix.copy_to_host()
    return cp.array(feature_matrix)


@nvtx.annotate("load_and_preprocess_data", color="blue")
def load_and_preprocess_data():
    print("Loading dataset...")
    DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
    DATASET_ENCODING = "ISO-8859-1"

    df = cudf.read_csv('sampled_dataset.csv', names=DATASET_COLUMNS, header=None)
    df = df[['text', 'target']].dropna().drop_duplicates()

    print(f"Number of rows loaded: {len(df)}")
    df = preprocess_target(df)
    return df


@nvtx.annotate("train_test_split", color="yellow")
def split_data(df):
    """
    Splits the dataset into training and testing sets and removes null values.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['target'], test_size=0.2, random_state=42
    )

    # Remove null values explicitly
    train_data = cudf.DataFrame({'text': X_train, 'target': y_train}).dropna()
    test_data = cudf.DataFrame({'text': X_test, 'target': y_test}).dropna()

    X_train, y_train = train_data['text'], train_data['target']
    X_test, y_test = test_data['text'], test_data['target']

    return X_train, X_test, y_train, y_test


@nvtx.annotate("train_model", color="cyan")
def train_model(X_train_features, y_train):
    """
    Train GPU-accelerated logistic regression model.
    """
    if cp.isnan(X_train_features).any():
        raise ValueError("X_train_features contains null values.")
    if y_train.isnull().any():
        raise ValueError("y_train contains null values.")

    model = LogisticRegression()
    model.fit(X_train_features, y_train)
    return model


@nvtx.annotate("evaluate_model", color="orange")
def evaluate_model(model, X_test_features, y_test):
    y_pred = model.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}")


@nvtx.annotate("main_pipeline", color="green")
def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Tokenizing texts on GPU...")
    X_train_features = tokenize_texts_gpu(X_train)
    X_test_features = tokenize_texts_gpu(X_test)

    print("Training GPU-accelerated model...")
    model = train_model(X_train_features, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test_features, y_test)
    


if __name__ == "__main__":
    main()
