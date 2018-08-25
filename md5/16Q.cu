// Memory object sizes:
// 1. hash table build: 2 * 8 * RLEN + 2 * 32 * 1024 * RBUCKETS
// 2. after hash_build before hash_join: 8 * RLEN
// 3. each hash_join: 8 * S_CHUNK_LEN + 8 * RLEN + 8 * n_results
#include <stdio.h>
#include <stdlib.h>

    #include "../md5/defs.h"
    #include "../md5/kernels.cu"

#include "hj.cu"
// #include "hj_kernels.cu"

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

//#define NR_BUCKETS_DEFAULT    256


// number of records in R
//#define RLEN      (40 * 1024L * 1024L)
#define RLEN        (10L * 1024L * 1024L)
// max of R's keys
#define RKEY_MAX    (1024 * 256)
// seed of R's keys
#define RKEY_SEED   1
// number of buckets for R's hash table; should not be larger than RKEY_MAX
#define RBUCKETS    (1024 * 8) // must be power of 2

// max of S's keys
#define SKEY_MAX    (1024 * 256)
// seed of S's keys
#define SKEY_SEED   2

// number of records in each chunk read from S
#define S_CHUNK_LEN     (64L * 1024L)
// how many chunks to be read from S
#define S_CHUNK_CNT     5

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

    #define N 8

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
            // printf("%s\n", "1-1");
                CUDA_SAFE_CALL(cudaMalloc((void **)Data->chr_data, sizeof(char) * DataSize));
                break;
            case dt_int:
            // printf("%s\n", "1-2");
                CUDA_SAFE_CALL(cudaMalloc((void **)Data->int_data, sizeof(int) * DataSize));
                break;
            case dt_flt:
            // printf("%s\n", "1-3");
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

int read_r(record_t *r_tmp, int *rlen)
{
    // cudaError_t res;

    // record_t *r_tmp = (record_t *)malloc(size_r);
    // if(!r_tmp) {
    //  fprintf(stderr, "malloc failed for R\n");
    //  return -1;
    // }
    // record_t *r_tmp;

    unsigned int seed = RKEY_SEED;
    for(int i = 0; i < RLEN; i++) {
        r_tmp[i].y = rand_r(&seed) % RKEY_MAX;
        r_tmp[i].x = i;
    }

    // *r = r_tmp;
    *rlen = RLEN;
    return 0;
}

// return the number of records actually read
int read_s(record_t *s, int slen, int skey_start)
{

    static unsigned int seed = SKEY_SEED;
    for(int i = 0; i < slen; i++) {
        s[i].y = rand_r(&seed) % (SKEY_MAX - skey_start) + skey_start;
        s[i].x = skey_start + i;
    }
    return slen;
}

// Assume R is the small table, upon which a hash table is built and kept in
// GPU memory. Assume S is the large table, for which data are fetched chunk
// by chunk, with one chunk, after another, joined with R.
// A problem with hash join is that, even though the joined results may be few,
// the number of buckets and sparse memory regions touched by the join may be
// plenty.
int main()
{
    cudaStream_t *stream = (cudaStream_t *) malloc(31 * sizeof(cudaStream_t));

    record_t *h_r[8], *h_s[8][S_CHUNK_CNT];
    hash_table_t ht_r[8];
    int rlen, slen[8][S_CHUNK_CNT];
    struct timeval t1, t2, t_start, t_end;

    gettimeofday(&t_start, NULL);

    printf("Time of starting hj: %lf \n", TVAL(t_start));

    // Create cuda stream
        for (int i = 0; i < 31; ++i)
            CUDA_SAFE_CALL( cudaStreamCreate(&stream[i]) );


    int build_hash_blocks = 64, build_hash_threads_per_block = 128;
    int scan_blocks = 512, scan_chunks;
    int scan_threads_per_block = 128;
    int scan_elems_per_block = 2 * scan_threads_per_block;
    int bytes_smem = sizeof(int) * scan_elems_per_block;


        int mm_block = 8;
        int width = 8 * 256;
        int height = width;

        // thread per block and block per grid for job n
        dim3 dimBlock[N],dimGrid[N];

        // dimBlock[0].x = 64, dimBlock[0].y = 1, dimBlock[0].z = 1;
        // dimGrid[0].x = 4096 / dimBlock[0].x, dimGrid[0].y = 1, dimGrid[0].z = 1;

    for (int i = 0; i < 4; ++i)
    {
        dimBlock[i*2].x = 32, dimBlock[i*2].y = 1, dimBlock[i*2].z = 1;
        dimGrid[i*2].x = 4096 / dimBlock[i*2].x, dimGrid[i*2].y = 1, dimGrid[i*2].z = 1;
        dimBlock[i*2+1].x = 8, dimBlock[i*2+1].y = 8, dimBlock[i*2+1].z = 1;
        dimGrid[i*2+1].x = width / dimBlock[i*2+1].x, dimGrid[i*2+1].y = height / dimBlock[i*2+1].y, dimGrid[i*2+1].z = 1;
    }


        // Declare vars for host data and results
        struct ElementAttr job[N];
        struct ElementSet h_data[N], h_result[N];

        // Declare vars for device data and results
        struct ElementSet d_data[N], d_result[N];

    for (int i = 0; i < 4; ++i)
    {
        // Set job attributes
        job[i*2].type = dt_chr, job[i*2].dataSize = BYTES_INPUT, job[i*2].resultSize = dimGrid[i*2].x * dimBlock[i*2].x * MD5_LEN / sizeof(char);
        job[i*2+1].type = dt_flt, job[i*2+1].dataSize = 2 * height * width, job[i*2+1].resultSize = dimGrid[i*2+1].x * dimBlock[i*2+1].x;
    }

        int n = N;

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
        }

        // init
        srand(2018);

    for (int j = 0; j < 4; ++j)
    {
        // initialize host data
        for (int i = 0; i < job[j*2].dataSize; i++)
            h_data[0].chr_data[j*2] = (unsigned char)(rand() % 256);

        randomInit(h_data[j*2+1].flt_data, job[j*2+1].dataSize);
    }

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


    // read r and build hash table
    gettimeofday(&t1, NULL);

for(int i = 0; i < n; i++){
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_r[i], sizeof(record_t) * RLEN));

    if(read_r(h_r[i], &rlen)) {
        fprintf(stderr, "failed to read r\n");
        return -1;
    }
}
    gettimeofday(&t2, NULL);
    printf("Time on reading R: %lf ms\n", TIME_DIFF(t1, t2));

    gettimeofday(&t1, NULL);

    
    // printf("Begin build_hash_table(r)\n");

    // varaibales for building hash table
    int *d_hist[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, *d_loc[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    record_t *d_r[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    int ret = 0;


    for(int i = 0; i < 8; i++){
        ht_r[i].n_buckets = RBUCKETS;
        ht_r[i].d_rec = NULL;
        ht_r[i].d_idx = NULL;
        ht_r[i].n_records = rlen;
        if(!ht_r[i].n_buckets) {
            ht_r[i].n_buckets = NR_BUCKETS_DEFAULT;
        }
    }

    // for scan
    int *d_sumbuf[8];   // the buffer used to store sum updates across subarrays
    int *h_sumbuf[8];
    int sum_tot[8], sum_delta[8];


for(int i = 0; i < n; i++){
    // step 1: partition the array into many subarrays,
    // each of which is scanned separately
    scan_chunks = build_hash_blocks * build_hash_threads_per_block * ht_r[i].n_buckets / scan_elems_per_block;
    scan_chunks += (build_hash_blocks * build_hash_threads_per_block * ht_r[i].n_buckets % scan_elems_per_block) ? 1 : 0;
}

for(int i = 0; i < 8; i++){
    // copy records to GPU device memory
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_r[i], rlen * sizeof(record_t)));

    // build histogram matrix to collect how many
    // records each thread generates in each bucket
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_hist[i],
            sizeof(int) * build_hash_blocks * build_hash_threads_per_block * ht_r[i].n_buckets));

    // prefix sum to get the offsets
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_loc[i],
            sizeof(int) * build_hash_blocks * build_hash_threads_per_block * ht_r[i].n_buckets));

    // build the hash table
    CUDA_SAFE_CALL(cudaMalloc((void **)&ht_r[i].d_rec, rlen * sizeof(record_t)));


    CUDA_SAFE_CALL(cudaMalloc((void **)&ht_r[i].d_idx, (ht_r[i].n_buckets + 1) * sizeof(int)));


        CUDA_SAFE_CALL(cudaMalloc((void **)&d_sumbuf[i], sizeof(int) * scan_chunks));

        // printf("scan: begin cudaMallocHost\n");
        CUDA_SAFE_CALL(cudaMallocHost((void**)&h_sumbuf[i], sizeof(int) * scan_chunks));
        // printf("scan: finish cudaMallocHost\n");
}


for(int i = 0; i < 8; i++){
    // printf("build_hash_table: begin cudaMemcpyAsync(r)\n");
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_r[i], h_r[i], rlen * sizeof(record_t), cudaMemcpyHostToDevice, stream[i]));
}
    // printf("build_hash_table: finish cudaMemcpyAsync(r)\n");

        for (int i = 0; i < n; ++i) {
            // printf("%s\n", "2-loop-copyHtoD");
            switch(job[i].type){
                case dt_chr:
                    // printf("%s\n", "2-1-0");
                    // CUDA_SAFE_CALL(cudaMemcpy(d_data[i].chr_data, h_data[i].chr_data, sizeof(char) * job[i].dataSize, cudaMemcpyHostToDevice));
                    // printf("%s\n", "2-1");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_data[i].chr_data, h_data[i].chr_data, sizeof(char) * job[i].dataSize, cudaMemcpyHostToDevice, stream[i+8]));
                    break;
                case dt_int:
                    // CUDA_SAFE_CALL(cudaMemcpy(d_data[i].int_data, h_data[i].int_data, sizeof(int) * job[i].dataSize, cudaMemcpyHostToDevice));
                    // printf("%s\n", "2-2");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_data[i].int_data, h_data[i].int_data, sizeof(int) * job[i].dataSize, cudaMemcpyHostToDevice, stream[i+8]));
                    break;
                case dt_flt:
                    // CUDA_SAFE_CALL(cudaMemcpy(d_data[i].flt_data, h_data[i].flt_data, sizeof(float) * job[i].dataSize, cudaMemcpyHostToDevice));
                    // printf("%s\n", "2-3");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_data[i].flt_data, h_data[i].flt_data, sizeof(float) * job[i].dataSize, cudaMemcpyHostToDevice, stream[i+8]));
                    break;
            }
        }

for(int i = 0; i < 8; i++){
    hash_build_hist<<<build_hash_blocks, build_hash_threads_per_block, 0, stream[i]>>>(d_hist[i], d_r[i], rlen,
            ht_r[i].n_buckets);
    // printf("build_hash_table: finish hash_build_hist\n");
    if(cudaStreamSynchronize(stream[i]) != cudaSuccess) {
        fprintf(stderr, "kernel failed at hash_build_hist\n");
        ret = -1;
        goto failed;
    }
}

        for (int i = 0; i < n; ++i) {
            // printf("%s\n", "3-loop-execute-kernel");
            switch(i){
                case 0:
                    // printf("%s\n", "3-1");
                    md5_kernel<<<dimGrid[i], dimBlock[i], 0, stream[i+8]>>>(d_data[i].chr_data, d_result[i].chr_data, job[i].dataSize);
                    CUDA_SAFE_CALL(cudaStreamSynchronize(stream[i+2]));
                    break;
                case 1:
                    // printf("%s\n", "3-2");
                    MatMulKernel<<<dimGrid[i], dimBlock[i], 0, stream[i+8]>>>(d_data[i].flt_data, d_result[i].flt_data, width);
                    break;
            }
        }

for(int i = 0; i < 8; i++){
    // printf("scan: begin prefix_sum\n");
    prefix_sum<<<scan_blocks, scan_threads_per_block, bytes_smem, stream[i]>>>(
            d_loc[i], d_sumbuf[i], d_hist[i], scan_chunks, build_hash_blocks * build_hash_threads_per_block * ht_r[i].n_buckets);
    // printf("scan: finish prefix_sum\n");
    // printf("scan: begin cudaThreadSynchronize\n");
    if(cudaStreamSynchronize(stream[i]) != cudaSuccess) {
        fprintf(stderr, "kernel failed at prefix_sum\n");
        goto failed;
    }
}


        for (int i = 0; i < n; ++i) {
            // printf("%s\n", "4-copy DtoH");
            switch(job[i].type){
                case dt_chr:
                    // CUDA_SAFE_CALL(cudaMemcpy(h_result[i].chr_data, d_result[i].chr_data, sizeof(char) * job[i].resultSize, cudaMemcpyDeviceToHost));
                    // printf("%s\n", "4-1");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(h_result[i].chr_data, d_result[i].chr_data, sizeof(char) * job[i].resultSize, cudaMemcpyDeviceToHost, stream[i+8]));
                    break;
                case dt_int:
                    // CUDA_SAFE_CALL(cudaMemcpy(h_result[i].int_data, d_result[i].int_data, sizeof(int) * job[i].resultSize, cudaMemcpyDeviceToHost));
                    // printf("%s\n", "4-2");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(h_result[i].int_data, d_result[i].int_data, sizeof(int) * job[i].resultSize, cudaMemcpyDeviceToHost, stream[i+8]));
                    break;
                case dt_flt:
                    // CUDA_SAFE_CALL(cudaMemcpy(h_result[i].flt_data, d_result[i].flt_data, sizeof(float) * job[i].resultSize, cudaMemcpyDeviceToHost));
                    // printf("%s\n", "4-3");
                    CUDA_SAFE_CALL(cudaMemcpyAsync(h_result[i].flt_data, d_result[i].flt_data, sizeof(float) * job[i].resultSize, cudaMemcpyDeviceToHost, stream[i+8]));
                    break;
            }
        }
    // printf("scan: finish cudaThreadSynchronize\n");

    // free(h_sumbuf);
    // cudaFree(d_sumbuf);

    // step 2: update all scanned subarrays to derive the final result
    // res = cudaMemcpy(h_sumbuf, d_sumbuf, sizeof(int) * nr_chunks,
    //         cudaMemcpyDeviceToHost);
for(int i = 0; i < 8; i++){
    // printf("scan: begin cudaMemcpyAsync\n");
    CUDA_SAFE_CALL(cudaMemcpyAsync(h_sumbuf[i], d_sumbuf[i], sizeof(int) * scan_chunks,
            cudaMemcpyDeviceToHost, stream[i]));
    // printf("scan: finish cudaMemcpyAsync\n");
}

for(int j = 0; j < 8; j++){ 
    sum_tot[j] = 0;
    sum_delta[j] = h_sumbuf[j][0];
    for(int i = 1; i < scan_chunks; i++) {
        sum_tot[j] += sum_delta[j];
        sum_delta[j] = h_sumbuf[j][i];
        h_sumbuf[j][i] = sum_tot[j];
    }
    h_sumbuf[j][0] = 0;
    sum_tot[j] += sum_delta[j];
}

for(int i = 0; i < 8; i++){
    // printf("scan: begin cudaMemcpyAsync\n");
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_sumbuf[i], h_sumbuf[i], sizeof(int) * scan_chunks,
            cudaMemcpyHostToDevice, stream[i]));
}
    // printf("scan: finish cudaMemcpyAsync\n");

for(int i = 0; i < 8; i++){
    // printf("scan: begin prefix_sum_update\n");
    prefix_sum_update<<<scan_blocks, scan_threads_per_block, 0, stream[i]>>>(d_loc[i], d_sumbuf[i],
            scan_chunks, build_hash_blocks * build_hash_threads_per_block * ht_r[i].n_buckets);
    // printf("scan: finish prefix_sum_update\n");
    // printf("scan: begin cudaThreadSynchronize\n");
    if(cudaStreamSynchronize(stream[i]) != cudaSuccess) {
        fprintf(stderr, "kernel failed at prefix_sum_update\n");
        goto failed;
    }
    // printf("scan: finish cudaThreadSynchronize\n");

    hash_build<<<build_hash_blocks, build_hash_threads_per_block, 0, stream[i]>>>(ht_r[i].d_rec, ht_r[i].d_idx,
            d_r[i], rlen, d_loc[i], ht_r[i].n_buckets);
    if(cudaStreamSynchronize(stream[i]) != cudaSuccess) {
        fprintf(stderr, "kernel failed at hash_build\n");
        ret = -1;
        goto failed;
    }
}
    goto finish;

failed:
    free_hash_table(&ht_r[0]);
    free_hash_table(&ht_r[1]);
    // printf("scan: free\n");
    cudaFree(h_sumbuf);
    // printf("scan: cudafree\n");
    cudaFree(d_sumbuf);

finish:
    CUDA_FREE(d_r);
    CUDA_FREE(d_hist);
    CUDA_FREE(d_loc);
    // printf("scan: free\n");
    cudaFree(h_sumbuf);
    // printf("scan: cudafree\n");
    cudaFree(d_sumbuf);


// printf("build_hash_table: finish scan\n");
    CUDA_FREE(d_hist);


    cudaFree(h_r);  // table R on the host is not needed any more

    gettimeofday(&t2, NULL);
    printf("Time on building hash table for R: %lf ms\n", TIME_DIFF(t1, t2));


    // for each chunk of s, join with r
    // h_s = (record_t *)malloc(sizeof(record_t) * S_CHUNK_LEN);
    // if(!h_s) {
    //  fprintf(stderr, "malloc failed for s\n");
    //  free_hash_table(&ht_r);
    //  return -1;
    // }

    gettimeofday(&t1, NULL);
for(int k = 0; k < 8; k++){
    for(int i = 0; i < S_CHUNK_CNT; i++){
        CUDA_SAFE_CALL(cudaMallocHost((void**)&h_s[k][i], sizeof(record_t) * S_CHUNK_LEN));
        slen[k][i] = read_s(h_s[k][i], S_CHUNK_LEN, 0);
    }
}
    gettimeofday(&t2, NULL);
    printf("Time on reading S: %lf ms ( %lf ms per join )\n", TIME_DIFF(t1, t2), TIME_DIFF(t1, t2)/S_CHUNK_CNT);


    record_t *h_z[4][S_CHUNK_CNT];
    int zlen[4][S_CHUNK_CNT];

    gettimeofday(&t1, NULL);
for(int j = 0; j < 4; j++){
    for(int i = 0; i < S_CHUNK_CNT; i++){
        CUDA_SAFE_CALL(cudaMallocHost((void**)&h_z[j][i], sizeof(record_t) * S_CHUNK_LEN));
        zlen[j][i] = read_s(h_z[j][i], S_CHUNK_LEN, 0);
    }
}
    gettimeofday(&t2, NULL);
    printf("Time on reading Z: %lf ms ( %lf ms per join )\n", TIME_DIFF(t1, t2), TIME_DIFF(t1, t2)/S_CHUNK_CNT);


    // The number of result records joined per chunk is approximately:
    // RLEN * S_CHUNK_LEN / max(RKEY_MAX, SKEY_MAX)
    gettimeofday(&t1, NULL);

for(int k = 0; k < 4; k++){
    for(int i = 0; i < S_CHUNK_CNT; i++) {
        // printf("%d\n", i);
        // join with r
        if(slen[2*k][i] > 0) {
            // printf("Begin hash_join\n");
            if(hash_join(NULL, NULL, &ht_r[2*k], h_s[2*k][i], slen[2*k][i], stream, (k*S_CHUNK_CNT +i)%32)) {
                fprintf(stderr, "hash join failed for the %dth chunk of S\n",
                        i);
                break;
            }
            // printf("Finish hash_join\n");
        }

        if(slen[2*k+1][i] > 0) {
            // printf("Begin hash_join\n");
            if(hash_join(NULL, NULL, &ht_r[2*k+1], h_s[2*k+1][i], slen[2*k+1][i], stream, ((k+1)*S_CHUNK_CNT +i)%32)) {
                fprintf(stderr, "hash join failed for the %dth chunk of S\n",
                        i);
                break;
            }
            for(int j = 0; j < S_CHUNK_CNT; j++) {
                if(zlen[k][j] > 0) {
                    if(hash_join(NULL, NULL, &ht_r[2*k+1], h_z[k][j], zlen[k][j], stream, ((k+1)*S_CHUNK_CNT +i)%32)) {
                        fprintf(stderr, "hash join failed for the %dth chunk of Z\n", j);
                        break;
                    }
                }
            }
        }
    }
}
    gettimeofday(&t2, NULL);
    printf("Time on hash join: %lf ms ( %lf ms per join )\n", TIME_DIFF(t1, t2), TIME_DIFF(t1, t2)/S_CHUNK_CNT);

    free_hash_table(&ht_r[0]);
    free_hash_table(&ht_r[1]);
    cudaFree(h_s);

    gettimeofday(&t_end, NULL);
    printf("Total time taken: %lf ms\n", TIME_DIFF(t_start, t_end));
    printf("Time of ending hj: %lf \n", TVAL(t_end));
    return 0;
}
