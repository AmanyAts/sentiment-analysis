#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>  // For rand()
#include <ctime>    // For seeding rand()

#define MAX_TOKENS 128
#define NUM_FEATURES 128

// CUDA kernel for tokenization
__global__ void tokenize(char* text, float* token_ids, int text_size, int num_texts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < text_size) {
        int text_idx = idx / MAX_TOKENS;
        int char_idx = idx % MAX_TOKENS;
        if (text_idx < num_texts && char_idx < MAX_TOKENS) {
            token_ids[idx] = (float)(text[idx]) / 255.0f;  // Normalize to range [0, 1]
        }
    }
}

// CUDA kernel for forward propagation (dense layer)
__global__ void dense_layer(float* inputs, float* weights, float* biases, float* outputs, int num_samples, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        for (int j = 0; j < num_features; j++) {
            outputs[idx * num_features + j] = biases[j];
            for (int k = 0; k < num_features; k++) {
                outputs[idx * num_features + j] += inputs[idx * num_features + k] * weights[j * num_features + k];
            }
            // Clamp outputs to prevent overflow in activation
            outputs[idx * num_features + j] = max(-10.0f, min(10.0f, outputs[idx * num_features + j]));
        }
    }
}

// CUDA kernel for sigmoid activation
__global__ void sigmoid_activation(float* outputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        outputs[idx] = 1.0f / (1.0f + expf(-outputs[idx]));
    }
}

// Function to load dataset and extract text and labels
std::vector<std::pair<std::string, int>> load_dataset_with_labels(const std::string& filename) {
    nvtxRangePushA("Load Dataset");
    std::ifstream file(filename);
    std::vector<std::pair<std::string, int>> dataset;
    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string target, ids, date, flag, user, text;

        // Assuming the dataset columns are: target, ids, date, flag, user, text
        if (std::getline(iss, target, ',') &&  // Extract target (label)
            std::getline(iss, ids, ',') &&
            std::getline(iss, date, ',') &&
            std::getline(iss, flag, ',') &&
            std::getline(iss, user, ',') &&
            std::getline(iss, text, ',')) {
            
            int label = std::stoi(target);
            if (label == 4) label = 1;  // Convert positive sentiment (4) to 1
            dataset.emplace_back(text, label);  // Pair text with its label
        }
    }
    nvtxRangePop();
    return dataset;
}


// Preprocess text data and labels
void preprocess_text(const std::vector<std::pair<std::string, int>>& dataset, char* text_array, int* labels, int max_tokens) {
    nvtxRangePushA("Preprocess Text");
    for (size_t i = 0; i < dataset.size(); i++) {
        const std::string& text = dataset[i].first;
        labels[i] = dataset[i].second;  // Extract labels
        std::string truncated_text = text.substr(0, max_tokens);
        std::copy(truncated_text.begin(), truncated_text.end(), text_array + i * max_tokens);
    }
    nvtxRangePop();
}

// Evaluate the model's accuracy
float evaluate_model(float* predictions, int* labels, int num_samples) {
    nvtxRangePushA("Evaluate Model");
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        int pred_label = predictions[i] > 0.5 ? 1 : 0;
        if (pred_label == labels[i]) correct++;
    }
    nvtxRangePop();
    return static_cast<float>(correct) / num_samples;
}

// Debugging: Print intermediate results
void debug_print_labels(const std::vector<int>& labels, int limit = 10) {
    std::cout << "Labels (first " << limit << "): ";
    for (int i = 0; i < std::min((int)labels.size(), limit); i++) {
        std::cout << labels[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    const int max_tokens = MAX_TOKENS;
    const int num_features = NUM_FEATURES;

    // Load dataset and dynamically determine the number of samples
    nvtxRangePushA("Main Function");
    auto dataset = load_dataset_with_labels("sampled_dataset.csv");
    int num_samples = dataset.size();  // Adjust based on dataset size

    // Allocate memory for labels
    std::vector<int> labels(num_samples);

    // Preprocess text
    char* text_array = new char[num_samples * max_tokens];
    preprocess_text(dataset, text_array, labels.data(), max_tokens);

    // Debug: Print the first few labels
    debug_print_labels(labels);

    // Allocate GPU memory for text data and token IDs
    nvtxRangePushA("Allocate GPU Memory");
    char* d_text_array;
    float* d_token_ids;
    cudaMalloc(&d_text_array, num_samples * max_tokens * sizeof(char));
    cudaMalloc(&d_token_ids, num_samples * max_tokens * sizeof(float));
    nvtxRangePop();

    cudaMemcpy(d_text_array, text_array, num_samples * max_tokens * sizeof(char), cudaMemcpyHostToDevice);

    // Tokenize text on GPU
    nvtxRangePushA("Tokenization Kernel");
    int blockSize = 256;
    int numBlocks = (num_samples * max_tokens + blockSize - 1) / blockSize;
    tokenize<<<numBlocks, blockSize>>>(d_text_array, d_token_ids, num_samples * max_tokens, num_samples);
    cudaDeviceSynchronize();
    nvtxRangePop();

    // Initialize weights and biases
    nvtxRangePushA("Dense Layer Preparation");
    srand(time(0));  // Seed random number generator
    std::vector<float> h_weights(num_features * num_features);
    std::vector<float> h_biases(num_features, 0.0f);
    std::vector<float> h_outputs(num_samples * num_features, 0.0f);

    for (auto& weight : h_weights) {
        weight = static_cast<float>(rand()) / RAND_MAX * 0.01f;  // Initialize small random weights
    }

    float* d_weights;
    float* d_biases;
    float* d_outputs;

    cudaMalloc(&d_weights, num_features * num_features * sizeof(float));
    cudaMalloc(&d_biases, num_features * sizeof(float));
    cudaMalloc(&d_outputs, num_samples * num_features * sizeof(float));

    cudaMemcpy(d_weights, h_weights.data(), num_features * num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases.data(), num_features * sizeof(float), cudaMemcpyHostToDevice);
    nvtxRangePop();

    // Dense layer execution
    nvtxRangePushA("Dense Layer Execution");
    dense_layer<<<numBlocks, blockSize>>>(d_token_ids, d_weights, d_biases, d_outputs, num_samples, num_features);
    cudaDeviceSynchronize();
    nvtxRangePop();

    // Apply sigmoid activation
    nvtxRangePushA("Sigmoid Activation Kernel");
    sigmoid_activation<<<numBlocks, blockSize>>>(d_outputs, num_samples * num_features);
    cudaDeviceSynchronize();
    nvtxRangePop();

    // Copy results back for debugging
    cudaMemcpy(h_outputs.data(), d_outputs, num_samples * num_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Evaluate model
    float accuracy = evaluate_model(h_outputs.data(), labels.data(), num_samples);
    std::cout << "Model Accuracy: " << accuracy * 100 << "%" << std::endl;

    // Free memory
    cudaFree(d_text_array);
    cudaFree(d_token_ids);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_outputs);
    delete[] text_array;

    nvtxRangePop(); // End of Main Function
    return 0;
}
