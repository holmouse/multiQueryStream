/*-----------
 *
 * distanceGlobal.cu
 *
 * This is the source file of a kernel to calculate total distances 
 *
 * of all points only using global memory.
 *
 * streamsOptBenchmark/distanceGlobal.cu
 *
 * By Hao Li
 *
 *------------
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

 // __global__ void gpu_global_distance(float *d_res, float *d_x, float *d_y, int samples)
 __global__ void gpu_global_distance(float *d_res, float *d_x, int samples)
{
	for(int l = 0; l < 1000; l++)
    {
	int idx1 = blockDim.x * blockIdx.x + threadIdx.x;
	int idx2;
	float distance = 0.0;
	for (idx2 = 0; idx2 < samples; idx2++)
		// distance += sqrt((d_x[idx1]-d_x[idx2])*(d_x[idx1]-d_x[idx2]) + (d_y[idx1]-d_y[idx2])*(d_y[idx1]-d_y[idx2]));
		distance += sqrt((d_x[idx1] - d_x[idx2]) * (d_x[idx1] - d_x[idx2]) 
			+ (d_x[samples + idx1] - d_x[samples + idx2]) * (d_x[samples + idx1] - d_x[samples + idx2]));
	d_res[idx1] = distance / samples;
	}
}

/* Do not modify this function
 * compute sum of the average distances
 */
// float compute_sum(const float *array, const int n)
// {
// 	int i;
// 	float sum = 0.0;

// 	for(i=0; i<n; i++)
// 		sum += array[i];
// 	return sum;
// }

// #define thread_per_block 128
// #define SAMPLES 100

/* Do not modify this function
 * Initializes the input arrays
 */
void init_data(float **x, float **y, float **r, int n)
{
	if( n < 1){
		fprintf(stderr, "#of_samples should be +ve\n");
		exit(0);
	}

	/* Allocate memory for the arrays */
	int i;
	*x = (float*) malloc( sizeof(float)*n );
	*y = (float*) malloc( sizeof(float)*n );
	*r = (float*) malloc( sizeof(float)*n );

	if( *x==NULL || *y==NULL || *r==NULL ){
		fprintf(stderr, "Memory allocation failed\n");
		exit(0);
	}

	/* Generate random points between 0 and 1 */
	for(i=0; i<n; i++){
		(*x)[i] = (float) rand() / RAND_MAX;
		(*y)[i] = (float) rand() / RAND_MAX;
	}
}

// int main(int argc, char **argv)
// {

// 	float *host_x, *host_y;
// 	float *host_result;
// 	float host_sum = 0.0;
// 	int samples = SAMPLES;

// 	init_data(&host_x, &host_y, &host_result, samples);

// 	float *d_x, *d_y, *d_res, *d_res2;
// 	int num_of_block = ceil(samples/thread_per_block);
// 	int size = sizeof(float) * samples;

// 	cudaMalloc((void**)&d_x, size);
// 	cudaMalloc((void**)&d_y, size);
// 	cudaMalloc((void**)&d_res, size);
// 	cudaMalloc((void**)&d_res2, size);
// 	cudaMemcpy(d_x, host_x, size, cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_y, host_y, size, cudaMemcpyHostToDevice);

// 	dim3 GridDim(num_of_block, 1, 1), BlockDim(thread_per_block, 1, 1);
// 	gpu_global_distance<<<GridDim, BlockDim>>>(d_res, d_x, d_y, samples);
// 	cudaMemcpy(host_result, d_res, size, cudaMemcpyDeviceToHost);
// 	host_sum = compute_sum(host_result, samples);
// 	printf("GPU Global Memory -- Result = %f\n", host_sum);

// //	gpu_shared_memory<<<GridDim, BlockDim>>>(d_res2, d_x, d_y, samples);
// //	cudaMemcpy(host_result, d_res2, size, cudaMemcpyDeviceToHost);
// //	host_sum = compute_sum(host_result,samples);
// //	printf("GPU Shared Memory -- Result = %f", host_sum);


// 	cudaFree(d_x);
// 	cudaFree(d_y);
// 	cudaFree(d_res);
// 	//cudaFree(d_res2);
// 	free( host_x );
// 	free( host_y );
// 	free( host_result );

// 	return 0;
// }
