#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024

__global__ void backsub(float *a, float *b, float *x, int N){

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int j;
    float sum;

    if(index < N){
        sum = 0.0;
        for (j = 0; j < index; j++)
            sum += (x[j] * a[index * N + j]);
		x[index] = (b[index] - sum) / a[index * N + index];
    }
    
}

int main ( int argc, char *argv[] )  {

	int   i, j, N;
	float *x, *b, *a;
    float *dx, *db, *da, sum;
	cudaEvent_t start, stop, startCopyHostToDevice, endCopyHostToDevice, startCopyDeviceToHost, endCopyDeviceToHost;
    float total_time = 0, copyHostToDevice_time = 0, copyDeviceToHost_time = 0;

	if (argc != 2) {printf ("Usage : %s <matrix size>\n", argv[0]);	exit(1);}

	N = strtol(argv[1], NULL, 10);

	// Allocate space for matrices
	a = (float*) malloc (N * N * sizeof(float));
	b = (float*) malloc (N * sizeof(float));
	x = (float*) malloc (N * sizeof(float));

	/* Create floats between 0 and 1. Diagonal elements between 2 and 3. */
	srand (time( NULL));
	for (i = 0; i < N; i++) {
		x[i] = 0.0;
		b[i] = (float)rand()/(RAND_MAX*2.0-1.0);
		a[i*i] = 2.0+(float)rand()/(RAND_MAX*2.0-1.0);
		for (j = 0; j < i; j++) 
			a[i*j] = (float)rand()/(RAND_MAX*2.0-1.0);
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
    cudaMalloc((void **)&da, N * N * sizeof(float));
    cudaMalloc((void **)&db, N * sizeof(float));
    cudaMalloc((void **)&dx, N * sizeof(float));

    // Copy values to the device variables
    cudaEventRecord(startCopyHostToDevice);

    cudaMemcpy(da, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(endCopyHostToDevice);
    cudaEventSynchronize(endCopyHostToDevice);

	// Launch the kernel: Backwards Substitution
    backsub<<<((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(da, db, dx, N);

    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

    // Copy result back to host
    cudaEventRecord(startCopyDeviceToHost);

    cudaMemcpy(x, dx, N * sizeof(float), cudaMemcpyDeviceToHost);

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
    cudaFree(dx);

	/* // Print result
	for (i = 0; i < N; i++) {
		for (j = 0; j <= i; j++)
			printf ("%f \t", a[i*j]);	
		printf ("%f \t%f\n", x[i], b[i]);
	}

    // Check result
	for (i = 0; i < N; i++) {
		sum = 0.0;
		for (j = 0; j <= i; j++) 
			sum = sum + (x[j]*a[i*N+j]);	
		if (fabsf(sum - b[i]) > 0.00001){
			printf("%f != %f\n", sum, b[i]);
			printf("Validation Failed...\n");
		}
	} */

    free(a);
    free(b);
    free(x);

    printf("Total time elapsed: %fs\n", total_time / 1000);
    printf("Copy host to device time: %fs\n", copyHostToDevice_time / 1000);
    printf("Copy device to host time: %fs\n", copyDeviceToHost_time / 1000);

    return 0;
}
