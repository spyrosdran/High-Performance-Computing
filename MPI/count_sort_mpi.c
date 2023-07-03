#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define UPPER 1000
#define LOWER 0

int main(int argc, char *argv[])
{
    int *x, *y, *reduced;
    int i, j, my_num, my_place, n;
    int rank, size;

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if(rank==0)
        if (argc != 2) {
            printf ("Usage : %s <array_size>\n", argv[0]);
            return 1;
        }

    n = strtol(argv[1], NULL, 10);
    x = (int*) malloc (n * sizeof(int));
    y = (int*) malloc (n * sizeof(int));

    if (rank == 0) {
        // Initialize the table which will be used for the reduction
		reduced = (int*) malloc (n * sizeof(int));
        for (i=0; i<n; i++){
            x[i] = n - i;
            reduced [i] = 0;
        }
    }

    for (i=0; i<n; i++)
        y[i] = 0;

    // Broadcasting the x table, containing the elements to be sorted
    MPI_Bcast(x, n, MPI_INT, 0, MPI_COMM_WORLD);

    int block_size = n / size;
    int remaining = n % size;

    // Every process fills its own y table
    for (int j = rank * block_size; j < ((rank+1) * block_size); j++) {

        my_num = x[j];
        my_place = 0;
        for (i = 0; i < n; i++)
            if ((my_num > x[i]) || ((my_num == x[i]) && (j < i)))
                my_place++;
        y[my_place] = my_num;

    }

    // Master checks and sorts the remaining elements
    if (remaining > 0 && rank == 0) {

        for (int j = n - remaining; j < n; j++) {

            my_num = x[j];
            my_place = 0;
            for (i = 0; i < n; i++)
                if ((my_num > x[i]) || ((my_num == x[i]) && (j < i)))
                    my_place++;
            y[my_place] = my_num;

        }
    }

    // Reduction of the y table containing the sorted elements into the reduced table
    MPI_Reduce(y, reduced, n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Master prints the elements that were sorted
    if (rank == 0)
		for (i=0; i<n; i++)
			printf("%d\n", reduced[i]);

    free(x);
    free(y);
    free(reduced);

    MPI_Finalize();

    return 0;
}
