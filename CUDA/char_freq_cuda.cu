#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 128
#define base 0
#define THREADS_PER_BLOCK 1024

__global__ void char_freq(char *buffer, int *freq, long text_size) {
    long index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < text_size)
        atomicAdd(&(freq[buffer[index] - base]), 1);
}

int main(int argc, char *argv[]) {

    FILE *pFile;
    long file_size;
    char *buffer, *dbuffer;
    char *filename;
    size_t result;
    int j, freq[N], *dfreq;
    cudaEvent_t start, stop, startCopyHostToDevice, endCopyHostToDevice, startCopyDeviceToHost, endCopyDeviceToHost;
    float total_time = 0, copyHostToDevice_time = 0, copyDeviceToHost_time = 0;

    // Read command line argument
    if (argc != 2) {
        printf("Usage : %s <file_name>\n", argv[0]);
        return 1;
    }

    // Open file
    filename = argv[1];
    pFile = fopen(filename, "rb");
    if (pFile == NULL) {
        printf("File error\n");
        return 2;
    }

    // Obtain file size:
    fseek(pFile, 0, SEEK_END);
    file_size = ftell(pFile);
    rewind(pFile);
    printf("file size is %ld\n", file_size);

    // Allocate memory on host to contain the file
    buffer = (char *)malloc(sizeof(char) * file_size);
    if (buffer == NULL) {
        printf("Memory error\n");
        return 3;
    }

    // Copy the file into the buffer
    result = fread(buffer, 1, file_size, pFile);
    if (result != file_size) {
        printf("Reading error\n");
        return 4;
    }

    // Initialize frequency array
    for (j = 0; j < N; j++)
        freq[j] = 0;

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCopyHostToDevice);
    cudaEventCreate(&endCopyHostToDevice);
    cudaEventCreate(&startCopyDeviceToHost);
    cudaEventCreate(&endCopyDeviceToHost);

    // Start timing
    cudaEventRecord(start);

    // Allocate memory on the device
    cudaMalloc((void **)&dfreq, N * sizeof(int));
    cudaMalloc((void **)&dbuffer, file_size * sizeof(char));

    // Copy buffer to the device
    cudaEventRecord(startCopyHostToDevice);

    cudaMemcpy(dbuffer, buffer, file_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dfreq, freq, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(endCopyHostToDevice);
    cudaEventSynchronize(endCopyHostToDevice);

    // Launch the kernel: Character Frequency
    char_freq<<<((file_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dbuffer, dfreq, file_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

    // Copy the computed frequencies
    cudaEventRecord(startCopyDeviceToHost);

    cudaMemcpy(freq, dfreq, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(endCopyDeviceToHost);
    cudaEventSynchronize(endCopyDeviceToHost);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);    

    // Compute times
    cudaEventElapsedTime(&total_time, start, stop);
    cudaEventElapsedTime(&copyHostToDevice_time, startCopyHostToDevice, endCopyHostToDevice);
    cudaEventElapsedTime(&copyDeviceToHost_time, startCopyDeviceToHost, endCopyDeviceToHost);

    // Free the device memory
    cudaFree(dfreq);
    cudaFree(dbuffer);

    for (j = 0; j < N; j++)
        printf("%d = %d\n", j + base, freq[j]);

    fclose(pFile);
    free(buffer);

    printf("Total time elapsed: %fs\n", total_time / 1000);
    printf("Copy host to device time: %fs\n", copyHostToDevice_time / 1000);
    printf("Copy device to host time: %fs\n", copyDeviceToHost_time / 1000);

    return 0;
}
