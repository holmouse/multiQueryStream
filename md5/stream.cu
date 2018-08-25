    #include <cuda.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <sys/time.h>

    #ifdef USE_MQX
    #include "mqx.h"
    #endif

    #include "../md5/defs.h"
    #include "../md5/kernels.cu"

    #define TVAL(t)         ((t).tv_sec * 1000.0 + (t).tv_usec / 1000.0)
    #define TDIFF(t1, t2)   (TVAL(t2) - TVAL(t1))

    #ifndef CUDA_SAFE_CALL
    #define CUDA_SAFE_CALL(call) \
        do { \
            cudaError_t err = call; \
            if(cudaSuccess != err) { \
                fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                        __FILE__, __LINE__, cudaGetErrorString(err)); \
                exit(EXIT_FAILURE); \
            } \
        } while (0)
    #endif

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

    // Allocates a matrix with random float entries.
    void randomInit(float* data, int size)
    {
        for (int i = 0; i < size; ++i)
            data[i] = rand() / (float)RAND_MAX;
    }


    //float multiplication kernel called by MatMul()
    __global__ void MatMulKernel(float *A, float *C, int Width)
    {
            // Each thread computes one element of C by accumulating results into Cvalue
            float Cvalue = 0;
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            for (int e = 0; e < Width; ++e)
                    Cvalue += A[row * Width + e] * A[Width * Width + e * Width + col - 1];
            C[row * Width + col] = Cvalue;
    }

    #define N 3

    enum DataType { dt_chr, dt_int, dt_flt };

    struct ElementAttr{
        enum DataType type;
        int dataSize;
        int resultSize;
    };

    struct ElementSet {
      union {
        unsigned char *chr_data;
        int           *int_data;
        float         *flt_data;   
      };
    };

    int mallocMemory(struct ElementSet *Data, enum DataType Type, int DataSize){
        switch(Type){
            case dt_chr:
                CUDA_SAFE_CALL( cudaMallocHost((void**)&Data->chr_data, sizeof(char) * DataSize) );      // host pinned
                // Data->chr_data = (unsigned char *)malloc(sizeof(char) * DataSize);
                break;
            case dt_int:
                CUDA_SAFE_CALL( cudaMallocHost((void**)&Data->int_data, sizeof(int) * DataSize) );      // host pinned
                // Data->int_data = (int *)malloc(sizeof(int) * DataSize);
                break;
            case dt_flt:
                CUDA_SAFE_CALL( cudaMallocHost((void**)&Data->flt_data, sizeof(float) * DataSize) );      // host pinned
                // Data->flt_data = (float *)malloc(sizeof(float) * DataSize);
                break;
        }
        return 0;
    }

    int mallocMemoryOnDevice(struct ElementSet *Data, enum DataType Type, int DataSize){
        switch(Type){
            case dt_chr:
            printf("%s\n", "1-1");
                CUDA_SAFE_CALL(cudaMalloc((void **)Data->chr_data, sizeof(char) * DataSize));
                break;
            case dt_int:
            printf("%s\n", "1-2");
                CUDA_SAFE_CALL(cudaMalloc((void **)Data->int_data, sizeof(int) * DataSize));
                break;
            case dt_flt:
            printf("%s\n", "1-3");
                CUDA_SAFE_CALL(cudaMalloc((void **)Data->flt_data, sizeof(float) * DataSize));
                break;
        }
        return 0;
    }

    int printElement(struct ElementSet Data, struct ElementAttr Job){
        switch(Job.type){
            case dt_chr:
                for (int j = 0; j < Job.dataSize; ++j)
                    printf("%c\t", Data.chr_data[j]);
                printf("\n");
                    break;
            case dt_int:
                for (int j = 0; j < Job.dataSize; ++j)
                    printf("%d\t", Data.int_data[j]);
                printf("\n");
                break;
            case dt_flt:
                for (int j = 0; j < Job.dataSize; ++j)
                    printf("%f\t", Data.flt_data[j]);
                printf("\n");
                break;
        }
        return 0;
    }

    int main()
    {
        int n = N - 1;
        cudaStream_t stream[N];

        struct timeval t1, t2;
        float pinned_time = 0;

        // int mm_block = 128;
        // int width = 16 * 128;
        // int height = width;

        int mm_block = 8;
        int width = 8 * 256;
        int height = width;

        // thread per block and block per grid for job n
        dim3 dimBlock[N],dimGrid[N];

        // dimBlock[0].x = 64, dimBlock[0].y = 1, dimBlock[0].z = 1;
        // dimGrid[0].x = 4096 / dimBlock[0].x, dimGrid[0].y = 1, dimGrid[0].z = 1;
        dimBlock[0].x = 32, dimBlock[0].y = 1, dimBlock[0].z = 1;
        dimGrid[0].x = 4096 / dimBlock[0].x, dimGrid[0].y = 1, dimGrid[0].z = 1;
        dimBlock[1].x = 8, dimBlock[1].y = 8, dimBlock[1].z = 1;
        dimGrid[1].x = width / dimBlock[1].x, dimGrid[1].y = height / dimBlock[1].y, dimGrid[1].z = 1;

        // Declare vars for host data and results
        struct ElementAttr job[N];
        struct ElementSet h_data[N], h_result[N];

        // Declare vars for device data and results
        struct ElementSet d_data[N], d_result[N];

        // Set job attributes
        job[0].type = dt_chr, job[0].dataSize = BYTES_INPUT, job[0].resultSize = dimGrid[0].x * dimBlock[0].x * MD5_LEN / sizeof(char);
        job[1].type = dt_flt, job[1].dataSize = 2 * height * width, job[1].resultSize = dimGrid[1].x * dimBlock[1].x;


        gettimeofday(&t1, NULL);
        // Allocate memory 
        for(int i = 0; i < n; i++){
            // printf("%s\n", "0-loop-allocateMem");
            switch(job[i].type){
                case dt_chr:
                    // printf("%s\n", "0-1");
                    CUDA_SAFE_CALL( cudaMallocHost((void**)&h_data[i].chr_data, sizeof(char) * job[i].dataSize) );      // host pinned
                    CUDA_SAFE_CALL( cudaMallocHost((void**)&h_result[i].chr_data, sizeof(char) * job[i].resultSize) );      // host pinned
                    // Data->chr_data = (unsigned char *)malloc(sizeof(char) * DataSize);
                    break;
                case dt_int:
                    // printf("%s\n", "0-2");
                    CUDA_SAFE_CALL( cudaMallocHost((void**)&h_data[i].int_data, sizeof(int) * job[i].dataSize) );      // host pinned
                    CUDA_SAFE_CALL( cudaMallocHost((void**)&h_result[i].int_data, sizeof(int) * job[i].resultSize) );      // host pinned
                    // Data->int_data = (int *)malloc(sizeof(int) * DataSize);
                    break;
                case dt_flt:
                    // printf("%s\n", "0-3");
                    CUDA_SAFE_CALL( cudaMallocHost((void**)&h_data[i].flt_data, sizeof(float) * job[i].dataSize) );      // host pinned
                    CUDA_SAFE_CALL( cudaMallocHost((void**)&h_result[i].flt_data, sizeof(float) * job[i].resultSize) );      // host pinned
                    // Data->flt_data = (float *)malloc(sizeof(float) * DataSize);
                    break;
            }

            // mallocMemory(&h_data[i], job[i].type, job[i].dataSize);
            // mallocMemory(&h_result[i], job[i].type, job[i].resultSize);
        }

        gettimeofday(&t2, NULL);
        pinned_time = TDIFF(t1, t2);
        printf("pinned_time: %f ms\n", pinned_time);


        // printf("%s\n", "0-end loop");

        // init
        srand(2018);

        // printf("%s\n", "0.5-init");

        // initialize host data
        for (int i = 0; i < job[0].dataSize; i++)
            h_data[0].chr_data[i] = (unsigned char)(rand() % 256);

            // printf("%s\n", "0.5-init-1");
        
        // for (int i = 0; i < job[1].dataSize; ++i)
        //     h_data[1].flt_data[i] = rand() / (float)RAND_MAX;

        randomInit(h_data[1].flt_data, job[1].dataSize);

            // printf("%s\n", "0.5-init-2");

        // begin timing
        gettimeofday(&t1, NULL);

        // Allocate memory 
        for(int i = 0; i < n; i++){
            // printf("%s\n", "1-loop-allocateDeviceMem");
                switch(job[i].type){
                case dt_chr:
                // printf("%s\n", "1-1");
                    CUDA_SAFE_CALL(cudaMalloc((void **)&d_data[i].chr_data, sizeof(char) * job[i].dataSize));
                    CUDA_SAFE_CALL(cudaMalloc((void **)&d_result[i].chr_data, sizeof(char) * job[i].resultSize));
                    break;
                case dt_int:
                // printf("%s\n", "1-2");
                    CUDA_SAFE_CALL(cudaMalloc((void **)&d_data[i].int_data, sizeof(int) * job[i].dataSize));
                    CUDA_SAFE_CALL(cudaMalloc((void **)&d_result[i].int_data, sizeof(int) * job[i].resultSize));
                    break;
                case dt_flt:
                // printf("%s\n", "1-3");
                    CUDA_SAFE_CALL(cudaMalloc((void **)&d_data[i].flt_data, sizeof(float) * job[i].dataSize));
                    CUDA_SAFE_CALL(cudaMalloc((void **)&d_result[i].flt_data, sizeof(float) * job[i].resultSize));
                    break;
            }
            // mallocMemoryOnDevice(&d_data[i], job[i].type, job[i].dataSize);
            // mallocMemoryOnDevice(&d_result[i], job[i].type, job[i].resultSize);
        }

        // printf("%s\n", "1-end-loop");

        // Create cuda stream
        for (int i = 0; i < N; ++i)
            CUDA_SAFE_CALL( cudaStreamCreate(&stream[i]) );

        // Copy data from host to device
        for (int i = 0; i < n; ++i) {
            // printf("%s\n", "2-loop-copyHtoD");
            switch(job[i].type){
                case dt_chr:
                    // printf("%s\n", "2-1-0");
                    // CUDA_SAFE_CALL(cudaMemcpy(d_data[i].chr_data, h_data[i].chr_data, sizeof(char) * job[i].dataSize, cudaMemcpyHostToDevice));
                    // printf("%s\n", "2-1");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_data[i].chr_data, h_data[i].chr_data, sizeof(char) * job[i].dataSize, cudaMemcpyHostToDevice, stream[i]));
                    break;
                case dt_int:
                    // CUDA_SAFE_CALL(cudaMemcpy(d_data[i].int_data, h_data[i].int_data, sizeof(int) * job[i].dataSize, cudaMemcpyHostToDevice));
                    // printf("%s\n", "2-2");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_data[i].int_data, h_data[i].int_data, sizeof(int) * job[i].dataSize, cudaMemcpyHostToDevice, stream[i]));
                    break;
                case dt_flt:
                    // CUDA_SAFE_CALL(cudaMemcpy(d_data[i].flt_data, h_data[i].flt_data, sizeof(float) * job[i].dataSize, cudaMemcpyHostToDevice));
                    // printf("%s\n", "2-3");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_data[i].flt_data, h_data[i].flt_data, sizeof(float) * job[i].dataSize, cudaMemcpyHostToDevice, stream[i]));
                    break;
            }
        }
        // printf("%s\n", "2-end-loop");

        for (int i = 0; i < n; ++i) {
            // printf("%s\n", "3-loop-execute-kernel");
            switch(i){
                case 0:
                    // printf("%s\n", "3-1");
                    md5_kernel<<<dimGrid[i], dimBlock[i], 0, stream[i]>>>(d_data[i].chr_data, d_result[i].chr_data, job[i].dataSize);
                    CUDA_SAFE_CALL(cudaThreadSynchronize());
                    break;
                case 1:
                    // printf("%s\n", "3-2");
                    MatMulKernel<<<dimGrid[i], dimBlock[i], 0, stream[i]>>>(d_data[i].flt_data, d_result[i].flt_data, width);
                    break;
            }
        }
        // printf("%s\n", "3-end loop");

        // Copy result back to host
        for (int i = 0; i < n; ++i) {
            // printf("%s\n", "4-copy DtoH");
            switch(job[i].type){
                case dt_chr:
                    // CUDA_SAFE_CALL(cudaMemcpy(h_result[i].chr_data, d_result[i].chr_data, sizeof(char) * job[i].resultSize, cudaMemcpyDeviceToHost));
                    // printf("%s\n", "4-1");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(h_result[i].chr_data, d_result[i].chr_data, sizeof(char) * job[i].resultSize, cudaMemcpyDeviceToHost, stream[i]));
                    break;
                case dt_int:
                    // CUDA_SAFE_CALL(cudaMemcpy(h_result[i].int_data, d_result[i].int_data, sizeof(int) * job[i].resultSize, cudaMemcpyDeviceToHost));
                    // printf("%s\n", "4-2");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(h_result[i].int_data, d_result[i].int_data, sizeof(int) * job[i].resultSize, cudaMemcpyDeviceToHost, stream[i]));
                    break;
                case dt_flt:
                    // CUDA_SAFE_CALL(cudaMemcpy(h_result[i].flt_data, d_result[i].flt_data, sizeof(float) * job[i].resultSize, cudaMemcpyDeviceToHost));
                    // printf("%s\n", "4-3");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(h_result[i].flt_data, d_result[i].flt_data, sizeof(float) * job[i].resultSize, cudaMemcpyDeviceToHost, stream[i]));
                    break;
            }
        }
        // printf("%s\n", "4-end loop");

        gettimeofday(&t2, NULL);
        printf("Computing took %f ms\n", TDIFF(t1, t2));

        for (int i = 0; i < n; ++i) {
            printf("%s\n", "5-cuda free");
            switch(job[i].type){
                case dt_chr:
                    printf("%s\n", "5-1");
                    CUDA_SAFE_CALL(cudaFree(d_data[i].chr_data));
                    printf("%s\n", "5-1-1");
                    CUDA_SAFE_CALL(cudaFree(d_result[i].chr_data));
                    break;
                case dt_int:
                    printf("%s\n", "5-2");
                    CUDA_SAFE_CALL(cudaFree(d_data[i].int_data));
                    CUDA_SAFE_CALL(cudaFree(d_result[i].int_data));
                    break;
                case dt_flt:
                    printf("%s\n", "5-3");
                    CUDA_SAFE_CALL(cudaFree(d_data[i].flt_data));
                    CUDA_SAFE_CALL(cudaFree(d_result[i].flt_data));
                    break;
            }
        }
        printf("%s\n", "5-end loop");

        for (int i = 0; i < N; ++i)
            CUDA_SAFE_CALL( cudaStreamDestroy(stream[i]) );

        // gettimeofday(&t2, NULL);
        // printf("Time of starting: %lf \n", TVAL(t2));
        // printf("Computing took %f ms\n", TDIFF(t1, t2));

        for (int i = 0; i < n; ++i) {
            switch(job[i].type){
                case dt_chr:
                    cudaFreeHost(h_data[i].chr_data);
                    cudaFreeHost(h_result[i].chr_data);
                    break;
                case dt_int:
                    cudaFreeHost(h_data[i].int_data);
                    cudaFreeHost(h_result[i].int_data);
                    break;
                case dt_flt:
                    cudaFreeHost(h_data[i].flt_data);
                    cudaFreeHost(h_result[i].flt_data);
                    break;
            }
        }

        return 0;
    }

