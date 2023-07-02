#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024

__global__ void string_matching(char* buffer, int* match, char* pattern, long match_size, long pattern_size, int* total_matches){

    int i;
    long index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < match_size){

        for (i = 0; i < pattern_size && pattern[i] == buffer[i + index]; ++i);

        if (i >= pattern_size){
            match[index] = 1;
            atomicAdd(total_matches, 1);
        }
            
    }    

}

int main (int argc, char *argv[]) {
	
    FILE *pFile;
    long file_size, match_size, pattern_size;
    char *buffer, *dbuffer;
    char *filename, *pattern, *dpattern;
    size_t result;
    int j, *match, *dmatch, total_matches, *dtotal_matches;
    cudaEvent_t start, stop, startCopyHostToDevice, endCopyHostToDevice, startCopyDeviceToHost, endCopyDeviceToHost;
    float total_time = 0, copyHostToDevice_time = 0, copyDeviceToHost_time = 0;

    if (argc != 3) {
        printf ("Usage : %s <file_name> <string>\n", argv[0]);
        return 1;
    }

    filename = argv[1];
    pattern = argv[2];

    pFile = fopen ( filename , "rb" );
    if (pFile==NULL) {printf ("File error\n"); return 2;}

    // Obtain file size:
    fseek (pFile , 0 , SEEK_END);
    file_size = ftell (pFile);
    rewind (pFile);
    printf("file size is %ld\n", file_size);

    // Allocate memory to contain the file:
    buffer = (char*) malloc (sizeof(char)*file_size);
    if (buffer == NULL) {printf ("Memory error\n"); return 3;}

    // Copy the file into the buffer:
    result = fread (buffer,1,file_size,pFile);
    if (result != file_size) {printf ("Reading error\n"); return 4;} 

    pattern_size = strlen(pattern);
    match_size = file_size - pattern_size + 1;

    // Create the match table
    match = (int *) malloc (sizeof(int)*match_size);
    if (match == NULL) {printf ("Malloc error\n"); return 5;}

    // Initialize the match table
    total_matches = 0;
    for (j = 0; j < match_size; j++)
        match[j]=0;

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCopyHostToDevice);
    cudaEventCreate(&endCopyHostToDevice);
    cudaEventCreate(&startCopyDeviceToHost);
    cudaEventCreate(&endCopyDeviceToHost);

    // Start timing
    cudaEventRecord(start);

    // Create device variables
    cudaMalloc((void **)&dbuffer, sizeof(char)*file_size);
    cudaMalloc((void **)&dmatch, sizeof(int)*match_size);
    cudaMalloc((void **)&dpattern, sizeof(char)*pattern_size);
    cudaMalloc((void **)&dtotal_matches, sizeof(int));

    // Copy values from host to device
    cudaEventRecord(startCopyHostToDevice);

    cudaMemcpy(dbuffer, buffer, file_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dmatch, match, match_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dpattern, pattern, pattern_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dtotal_matches, &total_matches, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(endCopyHostToDevice);
    cudaEventSynchronize(endCopyHostToDevice);

    // Launch the kernel: Brute Force String Matching
    string_matching<<<(file_size + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dbuffer, dmatch, dpattern, match_size, pattern_size, dtotal_matches);

    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

    // Copy values from device to host
    cudaEventRecord(startCopyDeviceToHost);

    cudaMemcpy(match, dmatch, match_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_matches, dtotal_matches, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(endCopyDeviceToHost);
    cudaEventSynchronize(endCopyDeviceToHost);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   

    // Compute times
    cudaEventElapsedTime(&total_time, start, stop);
    cudaEventElapsedTime(&copyHostToDevice_time, startCopyHostToDevice, endCopyHostToDevice);
    cudaEventElapsedTime(&copyDeviceToHost_time, startCopyDeviceToHost, endCopyDeviceToHost);

    // Free unused memory
    cudaFree(dbuffer);
    cudaFree(dmatch);
    cudaFree(dpattern);
    cudaFree(dtotal_matches);

    // Printing the output
    //for (j = 0; j < match_size; j++)
    //    printf("%d", match[j]);
    
    printf("\nTotal matches = %d\n", total_matches);

    fclose (pFile);
    free (buffer);
    free (match);

    printf("Total time elapsed: %fs\n", total_time / 1000);
    printf("Copy host to device time: %fs\n", copyHostToDevice_time / 1000);
    printf("Copy device to host time: %fs\n", copyDeviceToHost_time / 1000);

    return 0;
}
