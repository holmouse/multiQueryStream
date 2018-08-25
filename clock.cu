/*-----------
 *
 * clock.cu
 *
 * This is the source file of a kernel to measure time for each block.
 *
 * This kernel is from CUDA samples. clock.cu
 *
 * streamsOptBenchmark/clock.cu
 *
 * By Hao Li
 *
 *------------
 */

// This kernel computes a standard parallel reduction and evaluates the
// time it takes to do that for each block. The timing results are stored
// in device memory.
__global__ static void clockTimedReduction(const float *input, float *output, clock_t *timer)
{
    for(int l = 0; l < 100000; l++)
    {

    // __shared__ float shared[2 * blockDim.x];
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) timer[bid] = clock();

    // Copy input.
    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    // Perform reduction to find minimum.
    for (int d = blockDim.x; d > 0; d /= 2)
    {
        __syncthreads();

        if (tid < d)
        {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0)
            {
                shared[tid] = f1;
            }
        }
    }

    // Write result.
    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    if (tid == 0) timer[bid+gridDim.x] = clock();

    }
}

// int main(int argc, char **argv)
// {
//     printf("CUDA Clock sample\n");

//     // This will pick the best possible CUDA capable device
//     int dev = findCudaDevice(argc, (const char **)argv);

//     float *dinput = NULL;
//     float *doutput = NULL;
//     clock_t *dtimer = NULL;

//     clock_t timer[NUM_BLOCKS * 2];
//     float input[NUM_THREADS * 2];

//     for (int i = 0; i < NUM_THREADS * 2; i++)
//     {
//         input[i] = (float)i;
//     }

//     checkCudaErrors(cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2));
//     checkCudaErrors(cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS));
//     checkCudaErrors(cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));

//     checkCudaErrors(cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice));

//     clockTimedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 *NUM_THREADS>>>(dinput, doutput, dtimer);

//     checkCudaErrors(cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost));

//     checkCudaErrors(cudaFree(dinput));
//     checkCudaErrors(cudaFree(doutput));
//     checkCudaErrors(cudaFree(dtimer));


//     // Compute the difference between the last block end and the first block start.
//     clock_t minStart = timer[0];
//     clock_t maxEnd = timer[NUM_BLOCKS];

//     for (int i = 1; i < NUM_BLOCKS; i++)
//     {
//         minStart = timer[i] < minStart ? timer[i] : minStart;
//         maxEnd = timer[NUM_BLOCKS+i] > maxEnd ? timer[NUM_BLOCKS+i] : maxEnd;
//     }

//     printf("Total clocks = %d\n", (int)(maxEnd - minStart));


//     // cudaDeviceReset causes the driver to clean up all state. While
//     // not mandatory in normal operation, it is good practice.  It is also
//     // needed to ensure correct operation when the application is being
//     // profiled. Calling cudaDeviceReset causes all profile data to be
//     // flushed before the application exits
//     cudaDeviceReset();

//     return EXIT_SUCCESS;
// }
