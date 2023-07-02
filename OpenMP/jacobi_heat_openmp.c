#include <stdio.h>
#include <math.h>
#include <omp.h>

#define maxsize 50
#define iterations 100
#define row 20
#define col 20
#define start 100
#define accuracy 50

int main(int argc, char* argv[])
{
    int i, j, k;
    double table1[maxsize][maxsize], table2[maxsize][maxsize];
    double diff;

    // Initialize both tables

    for(i=0;i<maxsize;i++)
    for(j=0;j<maxsize;j++)
    {
        table1[i][j]=0;
        table2[i][j]=0;
    }

    omp_set_num_threads(12);

    double start_time = omp_get_wtime();


    // Repeat for each iteration
    for(k = 0; k < iterations; k++)
    {

        // Create a heat source
        table1[row][col] = start;

        // Difference initialization
        diff = 0.0;

        // Perform the calculations
        #pragma parallel omp for reduction(+:diff) collapse(2) schedule(auto)
        for(i=1;i<maxsize-1;i++)
            for(j=1;j<maxsize-1;j++) {
                table2[i][j] = 0.25 *(table1[i-1][j] + table1[i+1][j] + table1[i][j-1] + table1[i][j+1]);
                diff += (table2[i][j]-table1[i][j])*(table2[i][j]-table1[i][j]);
            }

        // Print result 
        /*
        for(i=0;i<maxsize;i++)
        {
            for(j=0;j<maxsize;j++)
                printf("%5.0f ",table2[i][j]);
            printf("\n");
        }
        printf("\n");
        */

        // Print difference and check convergence
        diff = sqrt(diff);
        printf("diff = %3.25f\n\n", diff);

        if (diff < accuracy) {
            printf ("\n\nConvergence in %d iterations\n\n", k);
            break;
        }

        // Copy new table to old table
        for(i=0;i<maxsize;i++)
            for(j=0;j<maxsize;j++)
                table1[i][j]=table2[i][j];
    }

    double finish_time = omp_get_wtime();
    double elapsed = finish_time - start_time;

    printf("Elapsed time = %f seconds\n", elapsed);

    free(table1);
    free(table2);

    return 0;
}
