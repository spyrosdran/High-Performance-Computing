#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

__global__ void mxm(float *a, float *b, float *c, int N){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = blockIdx.y, k;
    float sum;

    sum = 0.0;
    for ( k = 0; k < N; k++)
        sum = sum + a[i*N+k] * b[k*N+j];
    c[i*N+j]=sum;

}

int main ( int argc, char *argv[] ) {

    
    float *a, *b, *c, *da, *db, *dc;
    int i, j;
    int N;
    int THREADS_PER_BLOCK = 64;
  
    if (argc != 2) { printf ("Usage : %s <matrix size>\n", argv[0]); exit(1); }
    N = strtol(argv[1], NULL, 10);

    dim3 threads(THREADS_PER_BLOCK);
    dim3 grid((N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, N);
    
  	// Memory allocation
  	a = (float*) malloc(N * N * sizeof(float));
  	b = (float*) malloc(N * N * sizeof(float));
  	c = (float*) malloc(N * N * sizeof(float));

  	// Initialization
  	srand ( time ( NULL));

  	for ( i = 0; i < N; i++ ) 
        for (j = 0; j < N; j++ )
            //a[i*N+j] = (float) rand() / (RAND_MAX * 2.0 - 1.0);
            a[i*N+j] = 1.0;

	for ( i = 0; i < N; i++ ) 
        for (j = 0; j < N; j++ )
            //b[i*N+j] = (float) rand() / (RAND_MAX * 2.0 - 1.0);
            b[i*N+j] = 2.0;
            
    for ( i = 0; i < N; i++ ) 
        for (j = 0; j < N; j++ )
            c[i*N+j] = 0.0;

    // Allocate memory on the device
    cudaMalloc((void **)&da, sizeof(float) * N * N);
    cudaMalloc((void **)&db, sizeof(float) * N * N);
    cudaMalloc((void **)&dc, sizeof(float) * N * N);

    // Copy variables to the device
    cudaMemcpy(da, a, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(float) * N * N, cudaMemcpyHostToDevice);

  	// Computation
    mxm<<<grid, threads>>>(da, db, dc, N);

    // Copy result matrix to host
    cudaMemcpy(c, dc, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        
    // Free the unused device memory
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    /* // Print output
    for ( i = 0; i < N; i++ ) {
        for (j = 0; j < N; j++ )
            printf ("%1.3f\t", c[i*N+j]);
        printf("\n");
    } */
        
    free(a);
    free(b);
    free(c);

    return 0;
 
}

