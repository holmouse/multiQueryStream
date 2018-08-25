/*-----------
 *
 * distanceShared.cu
 *
 * This is the source file of a kernel to calculate total distances of all points using shared memory.
 *
 * streamsOptBenchmark/distanceShared.cu
 *
 * By Hao Li
 *
 *------------
 */

 // #include "distanceGlobal.cu"

 __device__ float get_distance(float x1, float y1, float x2, float y2)
{
	return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

// __global__ void gpu_shared_memory(float *d_res, float *d_x, float *d_y, int samples)
__global__ void gpu_shared_memory(float *d_res, float *d_x, int samples, int thread_per_block)
{
	for(int l = 0; l < 10000; l++)
    {
	int idx1 = blockDim.x * blockIdx.x + threadIdx.x;
	int idx2, i, j;
	float distance = 0.0;

	// __shared__ float l_x[thread_per_block];
	// __shared__ float l_y[thread_per_block];
	// __shared__ float r_x[thread_per_block];
	// __shared__ float r_y[thread_per_block];

	extern __shared__ float l_x[];
	// __shared__ float l_y[thread_per_block] = l_x[thread_per_block + ];
	// __shared__ float r_x[thread_per_block] = l_x[thread_per_block * 2 + ];
	// __shared__ float r_y[thread_per_block] = l_x[thread_per_block * 3 + ];

	if (idx1 < samples) {

    l_x[threadIdx.x] = d_x[idx1];
	// l_y[threadIdx.x] = d_y[idx1];
	l_x[thread_per_block + threadIdx.x] = d_x[samples + idx1];
	__syncthreads();

	for(idx2 = threadIdx.x + 1; idx2 < blockDim.x; idx2++)
	{
		// distance += get_distance(l_x[threadIdx.x], l_x[idx2], l_y[threadIdx.x], l_y[idx2]);
		distance += get_distance(l_x[threadIdx.x], l_x[idx2], 
			l_x[thread_per_block + threadIdx.x], l_x[thread_per_block + idx2]);
	}

	for(i = blockIdx.x + 1; i < ceilf(samples/blockDim.x);i++)
	{
		idx2 = blockDim.x * i + threadIdx.x;
		if (idx2 < samples)
	//	{
			// r_x[threadIdx.x] = d_x[idx2];
			l_x[thread_per_block * 2 + threadIdx.x] = d_x[idx2];
			// r_y[threadIdx.x] = d_y[idx2];
			l_x[thread_per_block * 3 + threadIdx.x] = d_x[samples + idx2];
			__syncthreads();

			for (j = 0; j < blockDim.x ;j++)
				// distance += get_distance(l_x[threadIdx.x], r_x[j], l_y[threadIdx.x], r_y[j]);
				distance += get_distance(l_x[threadIdx.x], l_x[thread_per_block * 2 + j],
					l_x[thread_per_block + threadIdx.x], l_x[thread_per_block * 3 + j]);

	//	}
	}
	d_res[idx1] = distance / (samples - idx1);
	}

	}

}

// int main(int argc, char **argv)
// {

// 	float *host_x, *host_y;
// 	float *host_result;
// 	float host_sum = 0.0;
// 	int samples = 502;
// 	int num_of_block = 14;

// 	init_data(&host_x, &host_y, &host_result, samples);

// 	float *d_x, *d_y, *d_res, *d_res2;
// 	int thread_per_block = ceil(samples/num_of_block);
// 	int size = sizeof(float) * samples;

// 	cudaMalloc((void**)&d_x, size);
// 	cudaMalloc((void**)&d_y, size);
// 	cudaMalloc((void**)&d_res, size);
// 	cudaMalloc((void**)&d_res2, size);
// 	cudaMemcpy(d_x, host_x, size, cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_y, host_y, size, cudaMemcpyHostToDevice);

// 	dim3 GridDim(num_of_block, 1, 1), BlockDim(thread_per_block, 1, 1);
// 	// gpu_avg_distance<<<GridDim, BlockDim>>>(d_res, d_x, d_y, samples);
// 	cudaMemcpy(host_result, d_res, size, cudaMemcpyDeviceToHost);
// 	// host_sum = compute_sum(host_result, samples);
// 	// printf("GPU Global Memory -- Result = %f", host_sum);

// 	cudaStream_t *streams = (cudaStream_t *) malloc(5 * sizeof(cudaStream_t));

//   for (int i = 0; i < (5 ); i++)
//   {
//      cudaStreamCreate(&(streams[i]));
//   }

// 	gpu_shared_memory<<<GridDim, BlockDim, sizeof(float) * 4 * 1024, streams[0]>>>(d_res2, d_x, samples, thread_per_block);
// 	gpu_shared_memory<<<GridDim, BlockDim, sizeof(float) * 4 * 1024, streams[1]>>>(d_res2, d_x, samples, thread_per_block);
// 	gpu_shared_memory<<<GridDim, BlockDim, sizeof(float) * 4 * 1024, streams[2]>>>(d_res2, d_x, samples, thread_per_block);
// 	gpu_shared_memory<<<GridDim, BlockDim, sizeof(float) * 4 * 1024, streams[3]>>>(d_res2, d_x, samples, thread_per_block);
// 	gpu_shared_memory<<<GridDim, BlockDim, sizeof(float) * 4 * 1024, streams[4]>>>(d_res2, d_x, samples, thread_per_block);
// 	cudaMemcpyAsync(host_result, d_res2, size, cudaMemcpyDeviceToHost, streams[0]);
// 	// cudaMemcpy(host_result, d_res2, size, cudaMemcpyDeviceToHost);
// 	//host_sum = compute_sum(host_result,samples);
// 	// printf("GPU Shared Memory -- Result = %f", host_sum);

// 	cudaFree(d_x);
// 	cudaFree(d_y);
// 	cudaFree(d_res);
// 	cudaFree(d_res2);
// 	free( host_x );
// 	free( host_y );
// 	free( host_result );

// 	return 0;
// }
