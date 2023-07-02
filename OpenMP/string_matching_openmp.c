#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int main (int argc, char *argv[]) {

	FILE *pFile;
	long file_size, match_size, pattern_size, total_matches;
	char * buffer;
	char * filename, *pattern;
	size_t result;
	int i, j, *match;

	if (argc != 3) {
		printf ("Usage : %s <file_name> <string>\n", argv[0]);
		return 1;
	}

	filename = argv[1];
	pattern = argv[2];

	pFile = fopen (filename, "rb");
	if (pFile==NULL) {printf ("File error\n"); return 2;}

	// Obtain file size
	fseek (pFile , 0 , SEEK_END);
	file_size = ftell (pFile);
	rewind (pFile);
	printf("file size is %ld\n", file_size);

	// Allocate memory to contain the file
	buffer = (char*) malloc (sizeof(char) * file_size);
	if (buffer == NULL) {printf ("Memory error\n"); return 3;}

	// Copy the file into the buffer
	result = fread (buffer,1,file_size,pFile);
	if (result != file_size) {printf ("Reading error\n"); return 4;}

	pattern_size = strlen(pattern);
	match_size = file_size - pattern_size + 1;

	match = (int *) malloc (sizeof(int) * match_size);
	if (match == NULL) {printf ("Malloc error\n"); return 5;}

	total_matches = 0;
	for (j = 0; j < match_size; j++){
		match[j]=0;
	}

	omp_set_num_threads(12);

	double start = omp_get_wtime();

    /* Brute Force string matching */
    #pragma omp parallel for reduction(+:total_matches) schedule(auto)
	for (j = 0; j < match_size; ++j) {
        for (i = 0; i < pattern_size && pattern[i] == buffer[i + j]; ++i);
        if (i >= pattern_size) {
            match[j] = 1;
            total_matches++;
            }
        }

    double finish = omp_get_wtime();
    double elapsed = finish - start;
    printf("Elapsed time = %f seconds\n", elapsed);


    /*
	for (j = 0; j < match_size; j++){
		printf("%d", match[j]);
	}
	*/

    printf("\nTotal matches = %ld\n", total_matches);

	fclose (pFile);
	free (buffer);
	free (match);

	return 0;
}
