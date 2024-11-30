import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
import nvtx  # NVTX for profiling annotations

def download_and_load_dataset():
    """Loads and preprocesses the Sentiment140 dataset."""
    with nvtx.annotate("Function: download_and_load_dataset", color="blue"):
        DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
        DATASET_ENCODING = "ISO-8859-1"
        
        # Load the dataset correctly
        df = pd.read_csv('sampled_dataset.csv', encoding=DATASET_ENCODING, header=0, names=DATASET_COLUMNS)

        # Preprocess the dataset
        data = df[['text', 'target']].copy()  # Create a copy to avoid SettingWithCopyWarning
        data['target'] = data['target'].replace(4, 1)  # Convert positive labels (4 -> 1)
        
        # Ensure `target` is of integer type
        data['target'] = data['target'].astype(int)
        
        # Remove duplicates and nulls
        data = data.drop_duplicates().dropna()

        return list(data['text']), list(data['target'])

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets with class balancing."""
    with nvtx.annotate("Function: split_data", color="green"):
        print("Checking class distribution...")
        class_counts = Counter(y)
        print(f"Class distribution before filtering: {class_counts}")

        # Filter out classes with fewer than 2 samples
        valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
        X_filtered = [x for x, label in zip(X, y) if label in valid_classes]
        y_filtered = [label for label in y if label in valid_classes]

        print("Splitting data into train/test sets...")
        return train_test_split(X_filtered, y_filtered, test_size=test_size, random_state=random_state, stratify=y_filtered)

def tokenize_data(X_train, X_test):
    """Tokenizes training and testing data using DistilBERT tokenizer."""
    with nvtx.annotate("Function: tokenize_data", color="yellow"):
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)
        return train_encodings, test_encodings

def create_tf_datasets(train_encodings, test_encodings, y_train, y_test, batch_size=32):
    """Creates TensorFlow datasets for training and testing."""
    with nvtx.annotate("Function: create_tf_datasets", color="orange"):
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            y_train
        )).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            y_test
        )).batch(batch_size)

        return train_dataset, test_dataset

def train_model(train_dataset, test_dataset, learning_rate=5e-5, epochs=3):
    """Trains a DistilBERT model for sentiment analysis."""
    with nvtx.annotate("Function: train_model", color="red"):
        # Load pre-trained DistilBERT model
        model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        # Train the model
        model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs
        )

        return model

def evaluate_model(model, test_dataset):
    """Evaluates the model on the test dataset."""
    with nvtx.annotate("Function: evaluate_model", color="purple"):
        results = model.evaluate(test_dataset)
        print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
        return results

def make_predictions(model, test_dataset):
    """Makes predictions on the test dataset."""
    with nvtx.annotate("Function: make_predictions", color="cyan"):
        predictions = model.predict(test_dataset)
        predicted_classes = np.argmax(predictions.logits, axis=-1)
        return predicted_classes

# Main Function to Execute the Workflow
def main():
        # Step 1: Load and preprocess the dataset
        print("Starting dataset download and preprocessing...")
        X, y = download_and_load_dataset()
        
        # Step 2: Split data into train/test sets
        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Step 3: Tokenize the data
        print("Tokenizing data...")
        train_encodings, test_encodings = tokenize_data(X_train, X_test)
        
        # Step 4: Prepare TensorFlow datasets
        print("Creating TensorFlow datasets...")
        train_dataset, test_dataset = create_tf_datasets(train_encodings, test_encodings, y_train, y_test)
        
        # Step 5: Train the model
        print("Training the model...")
        model = train_model(train_dataset, test_dataset)
        
        # Step 6: Evaluate the model
        print("Evaluating the model...")
        evaluate_model(model, test_dataset)
        
        # Step 7: Make predictions
        print("Making predictions...")
        predicted_classes = make_predictions(model, test_dataset)
        print("Predicted Classes (First 10):", predicted_classes[:10])

# Execute the main function
if __name__ == "__main__":
    main()
