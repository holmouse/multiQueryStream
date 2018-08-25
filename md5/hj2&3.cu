// Memory object sizes:
// 1. hash table build: 2 * 8 * RLEN + 2 * 32 * 1024 * RBUCKETS
// 2. after hash_build before hash_join: 8 * RLEN
// 3. each hash_join: 8 * S_CHUNK_LEN + 8 * RLEN + 8 * n_results
#include <stdio.h>
#include <stdlib.h>

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

//#define NR_BUCKETS_DEFAULT	256


// number of records in R
//#define RLEN		(40 * 1024L * 1024L)
#define RLEN		(10L * 1024L * 1024L)
// max of R's keys
#define RKEY_MAX	(1024 * 256)
// seed of R's keys
#define RKEY_SEED	1
// number of buckets for R's hash table; should not be larger than RKEY_MAX
#define RBUCKETS	(1024 * 8) // must be power of 2

// max of S's keys
#define SKEY_MAX	(1024 * 256)
// seed of S's keys
#define SKEY_SEED	2

// number of records in each chunk read from S
#define S_CHUNK_LEN		(64L * 1024L)
// how many chunks to be read from S
#define S_CHUNK_CNT		5

int read_r(record_t *r_tmp, int *rlen)
{
	// cudaError_t res;

	// record_t *r_tmp = (record_t *)malloc(size_r);
	// if(!r_tmp) {
	// 	fprintf(stderr, "malloc failed for R\n");
	// 	return -1;
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
	cudaStream_t *stream = (cudaStream_t *) malloc(15 * sizeof(cudaStream_t));

	record_t *h_r[2], *h_s[2][S_CHUNK_CNT];
	hash_table_t ht_r[2];
	int rlen, slen[2][S_CHUNK_CNT];
	struct timeval t1, t2, t_start, t_end;

	gettimeofday(&t_start, NULL);

	printf("Time of starting hj: %lf \n", TVAL(t_start));

	// Create cuda stream
        for (int i = 0; i < 15; ++i)
            CUDA_SAFE_CALL( cudaStreamCreate(&stream[i]) );


	int build_hash_blocks = 64, build_hash_threads_per_block = 128;
	int scan_blocks = 512, scan_chunks;
	int scan_threads_per_block = 128;
	int scan_elems_per_block = 2 * scan_threads_per_block;
	int bytes_smem = sizeof(int) * scan_elems_per_block;


	// read r and build hash table
	gettimeofday(&t1, NULL);

	CUDA_SAFE_CALL(cudaMallocHost((void**)&h_r[0], sizeof(record_t) * RLEN));
	CUDA_SAFE_CALL(cudaMallocHost((void**)&h_r[1], sizeof(record_t) * RLEN));

	if(read_r(h_r[0], &rlen)) {
		fprintf(stderr, "failed to read r\n");
		return -1;
	}

	if(read_r(h_r[1], &rlen)) {
		fprintf(stderr, "failed to read r\n");
		return -1;
	}

	gettimeofday(&t2, NULL);
	printf("Time on reading R: %lf ms\n", TIME_DIFF(t1, t2));

	gettimeofday(&t1, NULL);

	
	// printf("Begin build_hash_table(r)\n");

	// varaibales for building hash table
	int *d_hist[2] = {NULL, NULL}, *d_loc[2] = {NULL, NULL};
	record_t *d_r[2] = {NULL, NULL};
	int ret = 0;


	for(int i = 0; i < 2; i++){
		ht_r[i].n_buckets = RBUCKETS;
		ht_r[i].d_rec = NULL;
		ht_r[i].d_idx = NULL;
		ht_r[i].n_records = rlen;
		if(!ht_r[i].n_buckets) {
			ht_r[i].n_buckets = NR_BUCKETS_DEFAULT;
		}
	}

	// for scan
	int *d_sumbuf[2];	// the buffer used to store sum updates across subarrays
	int *h_sumbuf[2];
	int sum_tot[2], sum_delta[2];



	// step 1: partition the array into many subarrays,
	// each of which is scanned separately
	scan_chunks = build_hash_blocks * build_hash_threads_per_block * ht_r[0].n_buckets / scan_elems_per_block;
	scan_chunks += (build_hash_blocks * build_hash_threads_per_block * ht_r[0].n_buckets % scan_elems_per_block) ? 1 : 0;
	scan_chunks = build_hash_blocks * build_hash_threads_per_block * ht_r[1].n_buckets / scan_elems_per_block;
	scan_chunks += (build_hash_blocks * build_hash_threads_per_block * ht_r[1].n_buckets % scan_elems_per_block) ? 1 : 0;

for(int i = 0; i < 2; i++){
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


for(int i = 0; i < 2; i++){
    // printf("build_hash_table: begin cudaMemcpyAsync(r)\n");
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_r[i], h_r[i], rlen * sizeof(record_t), cudaMemcpyHostToDevice, stream[i]));
}
    // printf("build_hash_table: finish cudaMemcpyAsync(r)\n");

for(int i = 0; i < 2; i++){
	hash_build_hist<<<build_hash_blocks, build_hash_threads_per_block, 0, stream[i]>>>(d_hist[i], d_r[i], rlen,
	        ht_r[i].n_buckets);
	// printf("build_hash_table: finish hash_build_hist\n");
	if(cudaStreamSynchronize(stream[i]) != cudaSuccess) {
		fprintf(stderr, "kernel failed at hash_build_hist\n");
		ret = -1;
		goto failed;
	}
	

	// printf("build_hash_table: begin scan\n");


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
	// printf("scan: finish cudaThreadSynchronize\n");

	// free(h_sumbuf);
	// cudaFree(d_sumbuf);

	// step 2: update all scanned subarrays to derive the final result
	// res = cudaMemcpy(h_sumbuf, d_sumbuf, sizeof(int) * nr_chunks,
	//         cudaMemcpyDeviceToHost);
for(int i = 0; i < 2; i++){
	// printf("scan: begin cudaMemcpyAsync\n");
	CUDA_SAFE_CALL(cudaMemcpyAsync(h_sumbuf[i], d_sumbuf[i], sizeof(int) * scan_chunks,
	        cudaMemcpyDeviceToHost, stream[i]));
	// printf("scan: finish cudaMemcpyAsync\n");
}

for(int j = 0; j < 2; j++){	
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

for(int i = 0; i < 2; i++){
	// printf("scan: begin cudaMemcpyAsync\n");
	CUDA_SAFE_CALL(cudaMemcpyAsync(d_sumbuf[i], h_sumbuf[i], sizeof(int) * scan_chunks,
	        cudaMemcpyHostToDevice, stream[i]));
}
	// printf("scan: finish cudaMemcpyAsync\n");

for(int i = 0; i < 2; i++){
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


	cudaFree(h_r);	// table R on the host is not needed any more

	gettimeofday(&t2, NULL);
	printf("Time on building hash table for R: %lf ms\n", TIME_DIFF(t1, t2));


	// for each chunk of s, join with r
	// h_s = (record_t *)malloc(sizeof(record_t) * S_CHUNK_LEN);
	// if(!h_s) {
	// 	fprintf(stderr, "malloc failed for s\n");
	// 	free_hash_table(&ht_r);
	// 	return -1;
	// }

	gettimeofday(&t1, NULL);
for(int k = 0; k < 2; k++){
	for(int i = 0; i < S_CHUNK_CNT; i++){
		CUDA_SAFE_CALL(cudaMallocHost((void**)&h_s[k][i], sizeof(record_t) * S_CHUNK_LEN));
		slen[k][i] = read_s(h_s[k][i], S_CHUNK_LEN, 0);
	}
}
	gettimeofday(&t2, NULL);
	printf("Time on reading S: %lf ms ( %lf ms per join )\n", TIME_DIFF(t1, t2), TIME_DIFF(t1, t2)/S_CHUNK_CNT);


	record_t *h_z[S_CHUNK_CNT];
	int zlen[S_CHUNK_CNT];

	gettimeofday(&t1, NULL);
	for(int i = 0; i < S_CHUNK_CNT; i++){
		CUDA_SAFE_CALL(cudaMallocHost((void**)&h_z[i], sizeof(record_t) * S_CHUNK_LEN));
		zlen[i] = read_s(h_z[i], S_CHUNK_LEN, 0);
	}
	gettimeofday(&t2, NULL);
	printf("Time on reading Z: %lf ms ( %lf ms per join )\n", TIME_DIFF(t1, t2), TIME_DIFF(t1, t2)/S_CHUNK_CNT);


	// The number of result records joined per chunk is approximately:
	// RLEN * S_CHUNK_LEN / max(RKEY_MAX, SKEY_MAX)
	gettimeofday(&t1, NULL);

	for(int i = 0; i < S_CHUNK_CNT; i++) {
		// printf("%d\n", i);
		// join with r
		if(slen[0][i] > 0) {
			// printf("Begin hash_join\n");
			if(hash_join(NULL, NULL, &ht_r[0], h_s[0][i], slen[0][i], stream, i)) {
				fprintf(stderr, "hash join failed for the %dth chunk of S\n",
				        i);
				break;
			}
			// printf("Finish hash_join\n");
		}

		if(slen[1][i] > 0) {
			// printf("Begin hash_join\n");
			if(hash_join(NULL, NULL, &ht_r[1], h_s[1][i], slen[1][i], stream, S_CHUNK_CNT+i)) {
				fprintf(stderr, "hash join failed for the %dth chunk of S\n",
				        i);
				break;
			}
			for(int j = 0; j < S_CHUNK_CNT; j++) {
				if(zlen[j] > 0) {
					if(hash_join(NULL, NULL, &ht_r[1], h_z[j], zlen[j], stream, S_CHUNK_CNT+i)) {
						fprintf(stderr, "hash join failed for the %dth chunk of Z\n", j);
						break;
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
