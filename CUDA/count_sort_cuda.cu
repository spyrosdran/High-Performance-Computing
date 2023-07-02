#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define UPPER 1000
#define LOWER 0
#define BLOCKS 40
#define THREADS_PER_BLOCK 1024


__global__ void count_sort(int *x, int *y, int n) {

        int i, my_num, my_place;
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if(index < n) {
                my_num = x[index];
                my_place = 0;
                for (i=0; i<n; i++)
                        if ((my_num > x[i]) || (my_num == x[i])) 
                                my_place++;

                y[my_place] = my_num;
        }
       
}

int main(int argc, char *argv[])
{
    int *x, *y, *dx, *dy, i;
    int size;
    cudaEvent_t start, stop, startCopyHostToDevice, endCopyHostToDevice, startCopyDeviceToHost, endCopyDeviceToHost;
    float total_time = 0, copyHostToDevice_time = 0, copyDeviceToHost_time = 0;
    
    // Obtaining the array size
    if (argc != 2) {
            printf ("Usage : %s <array_size>\n", argv[0]);
            return 1;
    }
    
    // Allocate memory on host
    int n = strtol(argv[1], NULL, 10);
    x = ( int * ) malloc ( n * sizeof ( int ) );
    y = ( int * ) malloc ( n * sizeof ( int ) );

    // Initialize x array
    for (i=0; i<n; i++)
        x[i] = n - i;

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCopyHostToDevice);
    cudaEventCreate(&endCopyHostToDevice);
    cudaEventCreate(&startCopyDeviceToHost);
    cudaEventCreate(&endCopyDeviceToHost);

    // Start timing
    cudaEventRecord(start);

    //Allocate memory on device
    size = n * sizeof(int);
    cudaMalloc((void **)&dx, size);
    cudaMalloc((void **)&dy, size);

    // Copy the x array and n to the device
    cudaEventRecord(startCopyHostToDevice);

    cudaMemcpy(dx, x, size, cudaMemcpyHostToDevice);

    cudaEventRecord(endCopyHostToDevice);
    cudaEventSynchronize(endCopyHostToDevice);

    // Launch the kernel
    count_sort<<<((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(dx, dy, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

    // Copy the result array to the host
    cudaEventRecord(startCopyDeviceToHost);

    cudaMemcpy(y, dy, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(endCopyDeviceToHost);
    cudaEventSynchronize(endCopyDeviceToHost);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Compute times
    cudaEventElapsedTime(&total_time, start, stop);
    cudaEventElapsedTime(&copyHostToDevice_time, startCopyHostToDevice, endCopyHostToDevice);
    cudaEventElapsedTime(&copyDeviceToHost_time, startCopyDeviceToHost, endCopyDeviceToHost);

    // Freeing the device memory
    cudaFree(dy);
    cudaFree(dx);
            
    //for (i=0; i<n; i++) 
    //        printf("%d\n", y[i]);

    printf("Total time elapsed: %fs\n", total_time / 1000);
    printf("Copy host to device time: %fs\n", copyHostToDevice_time / 1000);
    printf("Copy device to host time: %fs\n", copyDeviceToHost_time / 1000);
                
    return 0;
}
