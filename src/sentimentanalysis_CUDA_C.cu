#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h> // Include for CUDA support

#define MAX_TOKENS 1024
#define NUM_FEATURES 1024

typedef struct {
    char text[MAX_TOKENS];
    int label;
} Post;

// Device version of custom_strlen (for GPU use)
__device__ int custom_strlen_device(const char *str) {
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    return length;
}

// Helper: Allocate memory and handle failure
void *safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        printf("Error: Memory allocation failed.\n");
        exit(1);
    }
    return ptr;
}

// Shuffle the dataset to randomize it
void shuffleDataset(Post *dataset, int num_samples) {
    srand(time(NULL));  // Seed for randomness
    for (int i = 0; i < num_samples; i++) {
        int j = rand() % num_samples;
        // Swap elements
        Post temp = dataset[i];
        dataset[i] = dataset[j];
        dataset[j] = temp;
    }
}

// CUDA kernel for tokenization and embedding (parallelized)
__global__ void tokenizeAndEmbedKernel(Post *dataset, float *token_ids, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        for (int j = 0; j < custom_strlen_device(dataset[idx].text); j++) {
            token_ids[idx * MAX_TOKENS + j] = (float)(dataset[idx].text[j]) / 255.0f;
        }
    }
}

// CUDA kernel for dense layer computation (parallelized)
__global__ void denseLayerKernel(float *inputs, float *weights, float *biases, float *outputs, int num_samples, int embedding_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        outputs[idx] = biases[0]; // Start with the bias
        for (int k = 0; k < embedding_size; k++) {
            outputs[idx] += inputs[idx * embedding_size + k] * weights[k]; // Only one output
        }
    }
}

// CUDA kernel for sigmoid activation (parallelized)
__global__ void sigmoidActivationKernel(float *outputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        outputs[idx] = 1.0f / (1.0f + expf(-outputs[idx]));
    }
}

// Memory allocation for GPU
void *cuda_malloc(size_t size) {
    void *ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    return ptr;
}

// Copy data from host to device
void cuda_memcpy_to_device(void *device_ptr, void *host_ptr, size_t size) {
    cudaError_t err = cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

// Copy data from device to host
void cuda_memcpy_to_host(void *host_ptr, void *device_ptr, size_t size) {
    cudaError_t err = cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

// Load dataset from file
int loadDataset(const char *filename, Post **dataset) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    size_t capacity = 1000;
    size_t count = 0;
    *dataset = (Post *)safe_malloc(capacity * sizeof(Post));

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        if (count >= capacity) {
            capacity *= 2;
            *dataset = (Post *)realloc(*dataset, capacity * sizeof(Post));
            if (!*dataset) {
                printf("Error: Memory reallocation failed.\n");
                fclose(file);
                exit(1);
            }
        }

        // Read label (first column of dataset)
        char *token = strtok(line, ",");
        if (!token) continue;

        (*dataset)[count].label = atoi(token);  // Use numeric labels (0 for negative, 4 for positive)

        // Skip unnecessary columns (3rd, 4th, and 5th columns)
        token = strtok(NULL, ",");
        for (int i = 0; i < 3; i++) {
            token = strtok(NULL, ",");
        }

        // Read the tweet content (last column)
        token = strtok(NULL, "\n");
        if (token) {
            strncpy((*dataset)[count].text, token, MAX_TOKENS - 1);
            (*dataset)[count].text[MAX_TOKENS - 1] = '\0';
            count++;
        }
    }

    fclose(file);
    return count;
}

// Load and split the dataset into training and testing
int loadAndSplitDataset(const char *filename, Post **trainSet, Post **testSet, int *trainSize, int *testSize) {
    clock_t start_time = clock(); // Start time measurement
    Post *dataset = NULL;
    int num_samples = loadDataset(filename, &dataset);  // Load entire dataset

    // Shuffle the dataset to randomize the order
    shuffleDataset(dataset, num_samples);

    // Split into 70% training and 30% testing
    *trainSize = (int)(num_samples * 0.7);
    *testSize = num_samples - *trainSize;

    // Allocate memory for the training and testing sets
    *trainSet = (Post *)safe_malloc(*trainSize * sizeof(Post));
    *testSet = (Post *)safe_malloc(*testSize * sizeof(Post));

    // Copy the data into the train and test sets
    for (int i = 0; i < *trainSize; i++) {
        (*trainSet)[i] = dataset[i];
    }
    for (int i = 0; i < *testSize; i++) {
        (*testSet)[i] = dataset[*trainSize + i];
    }

    free(dataset);  // Free the original dataset after splitting
    clock_t end_time = clock(); // End time measurement
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Loading and Splitting Time: %.4f seconds\n", execution_time);

    return num_samples;
}

// Modified tokenization using hash function (previously used ASCII values)
void tokenizeAndEmbed(Post *dataset, float *token_ids, int num_samples) {
    clock_t start_time = clock(); // Start time measurement

    int block_size = 256;  // Number of threads per block
    int grid_size = (num_samples + block_size - 1) / block_size;  // Number of blocks

    // Launch the kernel to tokenize and embed the dataset
    tokenizeAndEmbedKernel<<<grid_size, block_size>>>(dataset, token_ids, num_samples);

    // Wait for GPU to finish before measuring the time
    cudaDeviceSynchronize();

    clock_t end_time = clock(); // End time measurement
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Tokenization Time: %.4f seconds\n", execution_time);
}

// Random weight initialization using Xavier method
void init_weights(float *weights, int num_features) {
    clock_t start_time = clock(); // Start time measurement

    for (int i = 0; i < num_features * num_features; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * sqrt(2.0f / num_features); // Xavier initialization
    }

    clock_t end_time = clock(); // End time measurement
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Weight Initialization Time: %.4f seconds\n", execution_time);
}

// Dense layer computation
void denseLayer(float *inputs, float *weights, float *biases, float *outputs, int num_samples, int embedding_size) {
    clock_t start_time = clock(); // Start time measurement

    int block_size = 256;  // Number of threads per block
    int grid_size = (num_samples + block_size - 1) / block_size;  // Number of blocks

    // Launch the kernel to compute the dense layer
    denseLayerKernel<<<grid_size, block_size>>>(inputs, weights, biases, outputs, num_samples, embedding_size);

    // Wait for GPU to finish before measuring the time
    cudaDeviceSynchronize();

    clock_t end_time = clock(); // End time measurement
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Dense Layer Time: %.4f seconds\n", execution_time);
}

// Apply sigmoid activation
void sigmoidActivation(float *outputs, int size) {
    clock_t start_time = clock(); // Start time measurement

    int block_size = 256;  // Number of threads per block
    int grid_size = (size + block_size - 1) / block_size;  // Number of blocks

    // Launch the kernel to apply sigmoid activation
    sigmoidActivationKernel<<<grid_size, block_size>>>(outputs, size);

    // Wait for GPU to finish before measuring the time
    cudaDeviceSynchronize();

    clock_t end_time = clock(); // End time measurement
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Sigmoid Activation Time: %.4f seconds\n", execution_time);
}

// Evaluate model predictions
float evaluate(float *outputs, int *labels, int num_samples) {
    clock_t start_time = clock(); // Start time measurement

    int correct = 0;

    for (int i = 0; i < num_samples; i++) {
        int predicted_label = outputs[i] > 0.6f ? 4 : 0; // If > 0.6, predict positive (4), else negative (0)
        if (predicted_label == labels[i]) {
            correct++;
        }
    }

    clock_t end_time = clock(); // End time measurement
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Evaluation Time: %.4f seconds\n", execution_time);

    return (float)correct / num_samples;
}

int main() {
    clock_t start_time = clock(); // Start time measurement

    printf("Starting program...\n");

    Post *trainSet = NULL, *testSet = NULL;
    int trainSize, testSize;
    int num_samples = loadAndSplitDataset("training.1600000.processed.noemoticon.csv", &trainSet, &testSet, &trainSize, &testSize);

    if (trainSize == 0 || testSize == 0) {
        printf("Error: No samples found in dataset.\n");
        return 1;
    }

    printf("Loaded %d training samples and %d test samples.\n", trainSize, testSize);

    // Memory for the labels and token IDs for training
    int *trainLabels = (int *)safe_malloc(trainSize * sizeof(int));
    for (int i = 0; i < trainSize; i++) {
        trainLabels[i] = trainSet[i].label;
    }

    float *trainTokenIds = (float *)safe_malloc(trainSize * MAX_TOKENS * sizeof(float));
    float *trainWeights = (float *)safe_malloc(NUM_FEATURES * NUM_FEATURES * sizeof(float));
    float *trainBiases = (float *)safe_malloc(NUM_FEATURES * sizeof(float));
    float *trainOutputs = (float *)safe_malloc(trainSize * NUM_FEATURES * sizeof(float));

    srand(time(NULL));
    init_weights(trainWeights, NUM_FEATURES);

    for (int i = 0; i < NUM_FEATURES; i++) {
        trainBiases[i] = 0.0f;
    }

    // Tokenizing and embedding training dataset
    tokenizeAndEmbed(trainSet, trainTokenIds, trainSize);

    // Train the model with the training set
    denseLayer(trainTokenIds, trainWeights, trainBiases, trainOutputs, trainSize, NUM_FEATURES);

    // Apply sigmoid activation for training set
    sigmoidActivation(trainOutputs, trainSize);

    // Evaluate on the test set (after training)
    float accuracy = evaluate(trainOutputs, trainLabels, trainSize);

    // Now, evaluate on test set
    int *testLabels = (int *)safe_malloc(testSize * sizeof(int));
    for (int i = 0; i < testSize; i++) {
        testLabels[i] = testSet[i].label;
    }

    float *testTokenIds = (float *)safe_malloc(testSize * MAX_TOKENS * sizeof(float));
    float *testOutputs = (float *)safe_malloc(testSize * NUM_FEATURES * sizeof(float));

    // Tokenizing and embedding test dataset
    tokenizeAndEmbed(testSet, testTokenIds, testSize);

    // Use the trained model to make predictions on the test set
    denseLayer(testTokenIds, trainWeights, trainBiases, testOutputs, testSize, NUM_FEATURES);

    // Apply sigmoid activation for test set
    sigmoidActivation(testOutputs, testSize);

    // Evaluate the test set
    float testAccuracy = evaluate(testOutputs, testLabels, testSize);
    printf("Test set Accuracy: %.2f%%\n", testAccuracy * 100);

    clock_t end_time = clock(); // End time measurement
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Total Execution Time: %.4f seconds\n", execution_time);

    // Free memory
    free(trainSet);
    free(testSet);
    free(trainLabels);
    free(testLabels);
    free(trainTokenIds);
    free(testTokenIds);
    free(trainWeights);
    free(trainBiases);
    free(trainOutputs);
    free(testOutputs);

    return 0;
}
