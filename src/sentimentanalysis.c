#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <openacc.h>
#include <nvToolsExt.h> // NVTX for profiling annotations

#define MAX_TOKENS 128
#define NUM_FEATURES 128

typedef struct {
    char text[MAX_TOKENS];
    int label;
} Post;

// Custom implementation of strlen
int custom_strlen(const char *str) {
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    return length;
}

// Helper: Allocate memory and handle failure
void *safe_malloc(size_t size, const char *name) {
    void *ptr = malloc(size);
    if (!ptr) {
        printf("Error: Memory allocation failed for %s.\n", name);
        exit(1);
    }
    return ptr;
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
    *dataset = (Post *)safe_malloc(capacity * sizeof(Post), "dataset");

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

        char *token = strtok(line, ",");
        if (!token) continue;

        (*dataset)[count].label = atoi(token);

        token = strtok(NULL, ",");
        token = strtok(NULL, ",");
        token = strtok(NULL, ",");
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

// Tokenize and embed text data
void tokenizeAndEmbed(Post *dataset, float *token_ids, int num_samples) {
    nvtxRangePush("tokenizeAndEmbed");
    #pragma acc parallel loop collapse(2) present(dataset[:num_samples], token_ids[:num_samples * MAX_TOKENS])
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < MAX_TOKENS; j++) {
            token_ids[i * MAX_TOKENS + j] = (j < custom_strlen(dataset[i].text)) ? 
                ((float)(dataset[i].text[j]) / 255.0f) : 0.0f;
        }
    }
    nvtxRangePop();
}

// Dense layer computation
void denseLayer(float *inputs, float *weights, float *biases, float *outputs, int num_samples, int embedding_size) {
    nvtxRangePush("denseLayer");
    #pragma acc parallel loop collapse(2) present(inputs[:num_samples * embedding_size], weights[:embedding_size * embedding_size], biases[:embedding_size], outputs[:num_samples * embedding_size])
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < embedding_size; j++) {
            outputs[i * embedding_size + j] = biases[j];
            for (int k = 0; k < embedding_size; k++) {
                outputs[i * embedding_size + j] += inputs[i * embedding_size + k] * weights[j * embedding_size + k];
            }
            outputs[i * embedding_size + j] = fmaxf(-10.0f, fminf(10.0f, outputs[i * embedding_size + j]));
        }
    }
    nvtxRangePop();
}

// Apply sigmoid activation
void sigmoidActivation(float *outputs, int size) {
    nvtxRangePush("sigmoidActivation");
    #pragma acc parallel loop present(outputs[:size])
    for (int i = 0; i < size; i++) {
        outputs[i] = 1.0f / (1.0f + expf(-outputs[i]));
    }
    nvtxRangePop();
}

// Evaluate model predictions
float evaluate(float *outputs, int *labels, int num_samples) {
    nvtxRangePush("evaluate");
    int correct = 0;

    #pragma acc parallel loop reduction(+:correct) present(outputs[:num_samples], labels[:num_samples])
    for (int i = 0; i < num_samples; i++) {
        int predicted_label = outputs[i] > 0.5f ? 1 : 0;
        if (predicted_label == labels[i]) {
            correct++;
        }
    }

    nvtxRangePop();
    return (float)correct / num_samples;
}

int main() {
    printf("Starting program...\n");

    Post *dataset = NULL;
    int num_samples = loadDataset("last_100000_rows.csv", &dataset);

    int *labels = (int *)safe_malloc(num_samples * sizeof(int), "labels");
    for (int i = 0; i < num_samples; i++) {
        labels[i] = dataset[i].label;
    }

    float *token_ids = (float *)safe_malloc(num_samples * MAX_TOKENS * sizeof(float), "token_ids");
    float *weights = (float *)safe_malloc(NUM_FEATURES * NUM_FEATURES * sizeof(float), "weights");
    float *biases = (float *)safe_malloc(NUM_FEATURES * sizeof(float), "biases");
    float *outputs = (float *)safe_malloc(num_samples * NUM_FEATURES * sizeof(float), "outputs");

    for (int i = 0; i < NUM_FEATURES * NUM_FEATURES; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 0.01f;
    }
    for (int i = 0; i < NUM_FEATURES; i++) {
        biases[i] = 0.0f;
    }

    #pragma acc enter data copyin(dataset[:num_samples], labels[:num_samples], token_ids[:num_samples * MAX_TOKENS], weights[:NUM_FEATURES * NUM_FEATURES], biases[:NUM_FEATURES]) create(outputs[:num_samples * NUM_FEATURES])

    printf("Tokenizing and embedding dataset...\n");
    tokenizeAndEmbed(dataset, token_ids, num_samples);

    printf("Running dense layer...\n");
    denseLayer(token_ids, weights, biases, outputs, num_samples, NUM_FEATURES);

    printf("Applying sigmoid activation...\n");
    sigmoidActivation(outputs, num_samples);

    printf("Evaluating model...\n");
    float accuracy = evaluate(outputs, labels, num_samples);
    printf("Model Accuracy: %.2f%%\n", accuracy * 100);

    #pragma acc exit data delete(dataset[:num_samples], labels[:num_samples], token_ids[:num_samples * MAX_TOKENS], weights[:NUM_FEATURES * NUM_FEATURES], biases[:NUM_FEATURES], outputs[:num_samples * NUM_FEATURES])

    free(dataset);
    free(labels);
    free(token_ids);
    free(weights);
    free(biases);
    free(outputs);

    printf("Program completed.\n");
    return 0;
}
