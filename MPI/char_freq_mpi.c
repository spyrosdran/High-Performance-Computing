#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#define N 128
#define base 0

int main (int argc, char *argv[]) {

    FILE *pFile;
    long file_size;
    char *buffer;
    char *filename;
    size_t result;
    int i, j, freq[N], reduced[N], size, rank;
    int block_size, remaining;
    int *sendcounts, *displacements;
    char *text_slice;

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // Master reads the file
    if(rank==0)
    {
        // File name was not given in the arguments
        if (argc != 2){
            printf ("Usage : %s <file_name>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        filename = argv[1];
        pFile = fopen (filename , "rb");

        // File does not exist
        if (pFile==NULL){
            printf ("File error\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
            return 2;
        }

        // Obtain file size
        fseek (pFile , 0 , SEEK_END);
        file_size = ftell (pFile);
        rewind (pFile);
        printf("file size is %ld\n", file_size);

        // Allocate memory to contain the file
        buffer = (char*) malloc (sizeof(char) * file_size);

        if (buffer == NULL){
            printf ("Memory error\n");
            MPI_Abort(MPI_COMM_WORLD,3);
            return 3;
        }

        // Copy the file into the buffer
        result = fread (buffer,1,file_size,pFile);
        if (result != file_size){
            printf ("Reading error\n");
            MPI_Abort(MPI_COMM_WORLD,4);
            return 4;
        }

    }

    // Broadcasting file_size
    MPI_Bcast(&file_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // Initializing the frequency table
    for (int j=0; j<N; j++)
        freq[j]=0;

    // Defining block_size and see if there are remaining characters
    block_size = file_size / size;
    remaining = file_size % size;

    // Allocating memory for the text slice
    text_slice = (char*) malloc (sizeof(char)*block_size);
    
    // Allocating memory for the sendcounts and the displacements
    sendcounts = (int*) malloc(size * sizeof(int));
    displacements = (int*) malloc(size * sizeof(int));

    // Defining the sendcounts and the displacements
    for (int i = 0; i < size; i++){
        sendcounts[i] = block_size;
        displacements[i] = i * block_size;
    }    

    // Scattering the buffer with MPI_Scatterv
    MPI_Scatterv(buffer, sendcounts, displacements, MPI_CHAR, text_slice, block_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Every process creates its own frequency table of its own text slice
    for (int i=0; i<block_size; i++)
        freq[text_slice[i] - base]++;

    // Master checks the remaining part of the buffer
    if (remaining > 0 && rank == 0) {
        int start = file_size - remaining;
        
        for (int i = start; i < file_size; i++)
            freq[buffer[i] - base]++;
    }

    // Reduction of the frequency table
    MPI_Reduce(freq, reduced, N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Master prints the frequency of every character
    if(rank==0) {

        for (j=0; j<N; j++)
            printf("%d = %d\n", j+base, reduced[j]);

        fclose (pFile);
        free (buffer);           

    }

    MPI_Finalize();

    return 0;
}
