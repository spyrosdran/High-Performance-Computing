#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include "mpi.h"

int main (int argc, char *argv[]) {
	
	FILE *pFile;
	long file_size, match_size, pattern_size, total_matches, individual_total_matches, start, end;
	char *buffer;
	char *filename, *pattern;
	size_t result;
	int i, j, k, *match, *individual_match;
	int rank, size, individual_match_size;
	char *text_slice;
	int slice_size, remaining;
	int *sendcounts, *displacements;
	MPI_Status status;

	MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

	// Invalid arguments
	if (argc != 3) {
		printf ("Usage : %s <file_name> <string>\n", argv[0]);
		return 1;
	}

	filename = argv[1];
	pattern = argv[2];

	if (rank == 0) {
		
		pFile = fopen ( filename , "rb" );
		if (pFile==NULL) {printf ("File error\n"); return 2;}

		// Obtain file size
		fseek (pFile , 0 , SEEK_END);
		file_size = ftell (pFile);
		rewind (pFile);
		printf("file size is %ld\n", file_size);
		
		// Allocate memory to contain the file
		buffer = (char*) malloc (sizeof(char)*file_size);
		if (buffer == NULL) {printf ("Memory error\n"); return 3;}

		// Copy the file into the buffer:
		result = fread (buffer,1,file_size,pFile);
		if (result != file_size) {printf ("Reading error\n"); return 4;} 
		
		pattern_size = strlen(pattern);
		match_size = file_size - pattern_size + 1;
		
		match = (int*) malloc(sizeof(int) * match_size);
		if (match == NULL) {printf ("Malloc error\n"); return 5;}
		
		total_matches = 0;
		for (j = 0; j < match_size; j++)
			match[j]=0;

	}

	// Broadcasting values
	MPI_Bcast(&file_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);
	MPI_Bcast(&pattern_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(pattern, pattern_size, MPI_CHAR, 0, MPI_COMM_WORLD);

	// Defining the slice size
	slice_size = file_size / size;
	remaining = file_size % size;

	// Initializing individual variables
	individual_total_matches = 0;
	individual_match_size = slice_size - pattern_size + 1;
	individual_match = (int*) malloc(individual_match_size * sizeof(int));

	for (int i = 0; i < individual_match_size; i++)
		individual_match[i] = 0;

	// Allocating memory for the text slice
	text_slice = (char*) malloc(slice_size * sizeof(char));

	// Master defines the sendcounts and the displacements
	sendcounts = (int*) malloc(size * sizeof(int));
    displacements = (int*) malloc(size * sizeof(int));

    // Defining the sendcounts and the displacements
	for (int i = 0; i < size; i++){
		sendcounts[i] = slice_size;
		displacements[i] = i * slice_size;
	}

	// Scattering the buffer with MPI_Scatterv
	MPI_Scatterv(buffer, sendcounts, displacements, MPI_CHAR, text_slice, slice_size, MPI_CHAR, 0, MPI_COMM_WORLD);

	/* Brute Force string matching */
	for (j = 0; j < individual_match_size; ++j) {

		for (i = 0; i < pattern_size && pattern[i] == text_slice[i + j]; ++i);
		if (i >= pattern_size) {
			individual_match[j] = 1;
			individual_total_matches++;
		}

    }

	// Reduction of individual total matches to total matches
	MPI_Reduce(&individual_total_matches, &total_matches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	// Gathering the individual match tables
	MPI_Gatherv(individual_match, individual_match_size, MPI_INT, match, sendcounts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

	// Master checks the remaining part
	if (remaining >= pattern_size && rank == 0) {

		start = file_size - remaining;

		for (j = start; j < file_size; ++j) {

			for (i = 0; i < pattern_size && pattern[i] == buffer[i + j]; ++i);
			if (i >= pattern_size) {
				match[j] = 1;
				total_matches++;
			}

    	}
	}

	// Master checks the parts between slices
	if (rank == 0) {

		for (int i = 1; i < size; i++) {

			start = individual_match_size * i - 1;
			end = start + pattern_size;

			for (j = start; j < end; ++j) {
				for (k = 0; k < pattern_size && pattern[k] == buffer[k + j]; ++k);

				if (k >= pattern_size && match[j] != 1) {
					match[j] = 1;
					total_matches++;
				}
    		}
		}
	}

	// Printing output
	if (rank == 0){

		for (j = 0; j < match_size; j++)
		printf("%d", match[j]);
		printf("\nTotal matches = %ld\n", total_matches);

		fclose(pFile);

	}

	MPI_Finalize();

	return 0;
}
