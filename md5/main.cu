#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef USE_MQX
#include "mqx.h"
#endif

#include "defs.h"
#include "kernels.cu"

#define TVAL(t)         ((t).tv_sec * 1000.0 + (t).tv_usec / 1000.0)
#define TDIFF(t1, t2)   (TVAL(t2) - TVAL(t1))

static inline void cucheck(cudaError_t res, char *msg)
{
	if(res != cudaSuccess) {
		printf("%s failed\n", msg);
		exit(1);
	}
}

void save_data_to_disk(unsigned char *data, size_t size)
{
    FILE *fout = fopen("md5.output", "w");
    if (!fout) {
        perror("Failed to create output file");
        exit(1);
    }
    fwrite(data, sizeof(unsigned char), size, fout);
    fclose(fout);
}

int main()
{
	unsigned char *d_input, *d_output;
	unsigned char *input, *output;
	int threads_per_block = 64;
	int nr_blocks = 4096 / threads_per_block;
	size_t size_output = nr_blocks * threads_per_block * MD5_LEN;
	struct timeval t1, t2;

	gettimeofday(&t1, NULL);

	printf("Time of starting md5: %lf \n", TVAL(t1));

	input = (unsigned char *)malloc(BYTES_INPUT);
	if (!input) {
		perror("failed to allocate input array");
		exit(1);
	}
	srand(2014);
	for (int i = 0; i < BYTES_INPUT; i++)
		input[i] = (unsigned char)(rand() % 256);

	output = (unsigned char *)malloc(size_output);
	if (!output) {
	    perror("failed to allocate output array");
	    exit(1);
	}

	

	cucheck(cudaMalloc((void **)&d_input, BYTES_INPUT), "cudaMalloc");
	cucheck(cudaMalloc((void **)&d_output, size_output), "cudaMalloc");
	cucheck(cudaMemcpy(d_input, input, BYTES_INPUT, cudaMemcpyHostToDevice),
	        "cuMemcpyHtoD");

#ifdef USE_MQX
	cucheck(cudaAdvise(0, CADV_INPUT), "cudaAdvise");
	cucheck(cudaAdvise(1, CADV_OUTPUT), "cudaAdvise");
#endif
	md5_kernel<<<nr_blocks, threads_per_block>>>(d_input, d_output,
	        BYTES_INPUT);
	cucheck(cudaThreadSynchronize(), "cudaThreadSynchronize");

    cucheck(cudaMemcpy(output, d_output, size_output, cudaMemcpyDeviceToHost),
            "cudaMemcpy");

	// clean up
	cucheck(cudaFree(d_input), "cudaFree");
	cucheck(cudaFree(d_output), "cudaFree");

    gettimeofday(&t2, NULL);
    printf("Computing signature took %f ms\n", TDIFF(t1, t2));
    printf("Time of ending md5: %lf \n", TVAL(t2));

    //save_data_to_disk(output, size_output);

    free(output);
	free(input);
	return 0;
}
