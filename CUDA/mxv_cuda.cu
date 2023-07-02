# include <stdlib.h>
# include <stdio.h>
# include <time.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024

__global__ void multiply(int *a, int *b, int *c, int N){

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int j;

    if(index<N)
        for (j = 0; j < N; j++)
            c[index] += a[index*j] * b[j];

}

int main ( int argc, char *argv[] ) {
	
	int *a, *b, *c, *da, *db, *dc;
	int i, j, N;
    cudaEvent_t start, stop, startCopyHostToDevice, endCopyHostToDevice, startCopyDeviceToHost, endCopyDeviceToHost;
    float total_time = 0, copyHostToDevice_time = 0, copyDeviceToHost_time = 0;
 
	if (argc != 2) { printf ("Usage : %s <matrix size>\n", argv[0]); exit(1);}

	// Retrieving matrix size
	N = strtol(argv[1], NULL, 10);

	//Allocate memory for the matrices
    a = (int*) malloc (N * N * sizeof(int));	
	b = (int*) malloc (N * sizeof(int));
	c = (int*) malloc (N * sizeof(int));

	// Assign values to the matrices
	srand(time(NULL));

	for (i = 0; i < N * N; i++)
        a[i] = rand()%10;

	for ( i = 0; i < N; i++ ) {
	    b[i] = rand()%10;
		c[i] = 0;
    }

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCopyHostToDevice);
    cudaEventCreate(&endCopyHostToDevice);
    cudaEventCreate(&startCopyDeviceToHost);
    cudaEventCreate(&endCopyDeviceToHost);

    // Start timing
    cudaEventRecord(start);

    // Allocate memory for the device variables
    cudaMalloc((void **)&da, N * N * sizeof(int));
    cudaMalloc((void **)&db, N * sizeof(int));
    cudaMalloc((void **)&dc, N * sizeof(int));

    // Copy variables from host to device
    cudaEventRecord(startCopyHostToDevice);

    cudaMemcpy(da, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(endCopyHostToDevice);
    cudaEventSynchronize(endCopyHostToDevice);

	// Multiplication
	multiply<<<((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(da, db, dc, N);

    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

    // Copy the result vector from device to host
    cudaEventRecord(startCopyDeviceToHost);

    cudaMemcpy(c, dc, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(endCopyDeviceToHost);
    cudaEventSynchronize(endCopyDeviceToHost);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);    

    // Compute times
    cudaEventElapsedTime(&total_time, start, stop);
    cudaEventElapsedTime(&copyHostToDevice_time, startCopyHostToDevice, endCopyHostToDevice);
    cudaEventElapsedTime(&copyDeviceToHost_time, startCopyDeviceToHost, endCopyDeviceToHost);

    // Free the unused memory
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

	/* // Print the computed matrices
	for ( i = 0; i < N; i++ ) {
		for ( j = 0; j < N; j++ )
			printf ("%d ", a[i*j]); 
		printf("\t %d ", b[i]);
		printf("\t %d \n", c[i]);
	} */

    free(a);
    free(b);
    free(c);

    printf("Total time elapsed: %fs\n", total_time / 1000);
    printf("Copy host to device time: %fs\n", copyHostToDevice_time / 1000);
    printf("Copy device to host time: %fs\n", copyDeviceToHost_time / 1000);

    return 0;
}



