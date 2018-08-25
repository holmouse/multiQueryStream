#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "hj.h"
#include "hj_kernels.cu"

// default number of buckets chosen by build_hash_table
#define NR_BUCKETS_DEFAULT	256

// prefix sum. the sum of all elements in d_src is returned
// if successful; otherwise, -1 is returned.
int scan(int *d_dest, int *d_src, int len, cudaStream_t* Stream, int StreamID)
{
	cudaError_t res;
	int *d_sumbuf;	// the buffer used to store sum updates across subarrays
	int *h_sumbuf;
	int sum_tot, sum_delta;
	int nr_blocks = 512, nr_chunks;
	int nr_threads_per_block = 128;
	int nr_elems_per_block = 2 * nr_threads_per_block;
	int bytes_smem = sizeof(int) * nr_elems_per_block;

	// step 1: partition the array into many subarrays,
	// each of which is scanned separately
	nr_chunks = len / nr_elems_per_block;
	nr_chunks += (len % nr_elems_per_block) ? 1 : 0;

	res = cudaMalloc((void **)&d_sumbuf, sizeof(int) * nr_chunks);
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMemAlloc(&d_sumbuf) failed\n");
		return -1;
	}

	// h_sumbuf = (int *)malloc(sizeof(int) * nr_chunks);
	// if(!h_sumbuf) {
	// 	fprintf(stderr, "malloc() failed for h_sumbuf\n");
	// 	cudaFree(d_sumbuf);
	// 	return -1;
	// }

	// printf("scan: begin cudaMallocHost\n");
	res = cudaMallocHost((void**)&h_sumbuf, sizeof(int) * nr_chunks);
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMallocHost(&h_sumbuf) failed\n");
		return -1;
	}
	// printf("scan: finish cudaMallocHost\n");

	// printf("scan: begin prefix_sum\n");
	prefix_sum<<<nr_blocks, nr_threads_per_block, bytes_smem, Stream[StreamID]>>>(
	        d_dest, d_sumbuf, d_src, nr_chunks, len);
	// printf("scan: finish prefix_sum\n");
	// printf("scan: begin cudaThreadSynchronize\n");
	if(cudaThreadSynchronize() != cudaSuccess) {
		fprintf(stderr, "kernel failed at prefix_sum\n");
		free(h_sumbuf);
		cudaFree(d_sumbuf);
		return -1;
	}
	// printf("scan: finish cudaThreadSynchronize\n");

	// free(h_sumbuf);
	// cudaFree(d_sumbuf);

	// step 2: update all scanned subarrays to derive the final result
	// res = cudaMemcpy(h_sumbuf, d_sumbuf, sizeof(int) * nr_chunks,
	//         cudaMemcpyDeviceToHost);

	// printf("scan: begin cudaMemcpyAsync\n");
	res = cudaMemcpyAsync(h_sumbuf, d_sumbuf, sizeof(int) * nr_chunks,
	        cudaMemcpyDeviceToHost, Stream[StreamID]);
	// printf("scan: finish cudaMemcpyAsync\n");
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy DtoH failed\n");
		free(h_sumbuf);
		cudaFree(d_sumbuf);
		return -1;
	}

	sum_tot = 0;
	sum_delta = h_sumbuf[0];
	for(int i = 1; i < nr_chunks; i++) {
		sum_tot += sum_delta;
		sum_delta = h_sumbuf[i];
		h_sumbuf[i] = sum_tot;
	}
	h_sumbuf[0] = 0;
	sum_tot += sum_delta;

	// res = cudaMemcpy(d_sumbuf, h_sumbuf, sizeof(int) * nr_chunks,
	//         cudaMemcpyHostToDevice);
	// printf("scan: begin cudaMemcpyAsync\n");
	res = cudaMemcpyAsync(d_sumbuf, h_sumbuf, sizeof(int) * nr_chunks,
	        cudaMemcpyHostToDevice, Stream[StreamID]);
	// printf("scan: finish cudaMemcpyAsync\n");
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy HtoD failed\n");
		free(h_sumbuf);
		cudaFree(d_sumbuf);
		return -1;
	}

	// printf("scan: begin prefix_sum_update\n");
	prefix_sum_update<<<nr_blocks, nr_threads_per_block, 0, Stream[StreamID]>>>(d_dest, d_sumbuf,
	        nr_chunks, len);
	// printf("scan: finish prefix_sum_update\n");
	// printf("scan: begin cudaThreadSynchronize\n");
	if(cudaThreadSynchronize() != cudaSuccess) {
		fprintf(stderr, "kernel failed at prefix_sum_update\n");
		free(h_sumbuf);
		cudaFree(d_sumbuf);
		return -1;
	}
	// printf("scan: finish cudaThreadSynchronize\n");

	// printf("scan: free\n");
	cudaFree(h_sumbuf);
	// printf("scan: cudafree\n");
	cudaFree(d_sumbuf);
	return sum_tot;
}

int build_hash_table(hash_table_t *ht, record_t *h_r, int rlen, cudaStream_t* Stream, int StreamID)
{
	int nr_blocks = 64, nr_threads_per_block = 128;
	int *d_hist = NULL, *d_loc = NULL;
	record_t *d_r = NULL;
	cudaError_t res;
	int ret = 0;

	ht->d_rec = NULL;
	ht->d_idx = NULL;
	ht->n_records = rlen;
	if(!ht->n_buckets) {
		ht->n_buckets = NR_BUCKETS_DEFAULT;
	}

	// copy records to GPU device memory
	res = cudaMalloc((void **)&d_r, rlen * sizeof(record_t));
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(&d_r) failed\n");
		ret = -1;
		goto failed;
	}

    // res = cudaMemcpy(d_r, h_r, rlen * sizeof(record_t), cudaMemcpyHostToDevice);
    // printf("build_hash_table: begin cudaMemcpyAsync(r)\n");
    res = cudaMemcpyAsync(d_r, h_r, rlen * sizeof(record_t), cudaMemcpyHostToDevice, Stream[StreamID]);
    if(res != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy(r) failed\n");
 		ret = -1;
		goto failed;
    }

    // printf("build_hash_table: finish cudaMemcpyAsync(r)\n");

	// build histogram matrix to collect how many
    // records each thread generates in each bucket
	res = cudaMalloc((void **)&d_hist,
	        sizeof(int) * nr_blocks * nr_threads_per_block * ht->n_buckets);
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(&d_hist) failed\n");
		ret = -1;
		goto failed;
	}

	hash_build_hist<<<nr_blocks, nr_threads_per_block, 0, Stream[StreamID]>>>(d_hist, d_r, rlen,
	        ht->n_buckets);
	// printf("build_hash_table: finish hash_build_hist\n");
	if(cudaThreadSynchronize() != cudaSuccess) {
		fprintf(stderr, "kernel failed at hash_build_hist\n");
		ret = -1;
		goto failed;
	}

	// prefix sum to get the offsets
	res = cudaMalloc((void **)&d_loc,
	        sizeof(int) * nr_blocks * nr_threads_per_block * ht->n_buckets);
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(&d_loc) failed\n");
		ret = -1;
		goto failed;
	}

	// printf("build_hash_table: begin scan\n");
	if(scan(d_loc, d_hist, nr_blocks * nr_threads_per_block * ht->n_buckets, Stream, StreamID)
	        < 0) {
		fprintf(stderr, "scan failed\n");
		ret = -1;
		goto failed;
	}
	// printf("build_hash_table: finish scan\n");
	CUDA_FREE(d_hist);
	d_hist = NULL;

	// build the hash table
	res = cudaMalloc((void **)&ht->d_rec, rlen * sizeof(record_t));
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(&ht->d_rec) failed\n");
		ret = -1;
		goto failed;
	}

	res = cudaMalloc((void **)&ht->d_idx, (ht->n_buckets + 1) * sizeof(int));
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(&ht->d_idx) failed\n");
		ret = -1;
		goto failed;
	}


	hash_build<<<nr_blocks, nr_threads_per_block, 0, Stream[StreamID]>>>(ht->d_rec, ht->d_idx,
	        d_r, rlen, d_loc, ht->n_buckets);
	if(cudaThreadSynchronize() != cudaSuccess) {
		fprintf(stderr, "kernel failed at hash_build\n");
		ret = -1;
		goto failed;
	}

	goto finish;

failed:
	free_hash_table(ht);

finish:
	CUDA_FREE(d_r);
	CUDA_FREE(d_hist);
	CUDA_FREE(d_loc);
	return ret;
}

void free_hash_table(hash_table_t *ht)
{
	CUDA_FREE(ht->d_rec);
	CUDA_FREE(ht->d_idx);
	ht->d_rec = NULL;
	ht->d_idx = NULL;
	ht->n_records = 0;
	ht->n_buckets = 0;
}

int hash_join(record_t **h_res, int *reslen,
        hash_table_t *ht_r, record_t *h_s, int slen, cudaStream_t* Stream, int StreamID)
{
	cudaError_t res;
	int ret = 0, n_results;
	record_t *restmp = NULL;
	int nr_blocks = 256, nr_threads_per_block = 128;
	int *d_hist = NULL, *d_loc = NULL;
	record_t *d_s = NULL, *d_res = NULL;

	// copy S to GPU device memory
	res = cudaMalloc((void **)&d_s, slen * sizeof(record_t));
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(&d_s) failed\n");
		ret = -1;
		goto failed;
	}

	// printf("hash_join: begin cudaMemcpyAsync\n");
    // res = cudaMemcpy(d_s, h_s, slen * sizeof(record_t), cudaMemcpyHostToDevice);
    res = cudaMemcpyAsync(d_s, h_s, slen * sizeof(record_t), cudaMemcpyHostToDevice, Stream[StreamID]);
    // printf("hash_join: finish cudaMemcpyAsync\n");
    if(res != cudaSuccess) {
        fprintf(stderr, "cuMemcpyHtoD(s) failed\n");
 		ret = -1;
       goto failed;
    }

	// count the number of records joined by each thread
	res = cudaMalloc((void **)&d_hist,
	        sizeof(int) * nr_blocks * nr_threads_per_block);
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(&d_hist) failed\n");
		ret = -1;
		goto failed;
	}

	// printf("hash_join: begin hash_join_hist\n");
	hash_join_hist<<<nr_blocks, nr_threads_per_block, 0, Stream[StreamID]>>>(d_hist, ht_r->d_rec,
	        ht_r->d_idx, ht_r->n_buckets, d_s, slen);
	// printf("hash_join: finish hash_join_hist\n");
	if(cudaThreadSynchronize() != cudaSuccess) {
		fprintf(stderr, "kernel failed at hash_join_hist\n");
		ret = -1;
		goto failed;
	}

	// prefix sum to get the locations
	res = cudaMalloc((void **)&d_loc,
	        sizeof(int) * nr_blocks * nr_threads_per_block);
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(&d_loc) failed\n");
		ret = -1;
		goto failed;
	}

	// printf("hash_join: begin scan\n");
	n_results = scan(d_loc, d_hist, nr_blocks * nr_threads_per_block, Stream, StreamID);
	if(n_results < 0) {
		fprintf(stderr, "scan failed\n");
		ret = -1;
		goto failed;
	}
	// printf("hash_join: finish scan\n");
	CUDA_FREE(d_hist);
	d_hist = NULL;

	if(n_results <= 0) {
		if(h_res) {
			*h_res = NULL;
		}

		if(reslen) {
			*reslen = 0;
		}

		goto finish;
	}

	// do hash join
	res = cudaMalloc((void **)&d_res, n_results * sizeof(record_t));
	if(res != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(&d_res) failed\n");
		ret = -1;
		goto failed;
	}

	// printf("hash_join: begin hash_join\n");
	hash_join<<<nr_blocks, nr_threads_per_block, 0, Stream[StreamID]>>>(d_res, d_loc, ht_r->d_rec,
	        ht_r->d_idx, ht_r->n_buckets, d_s, slen);
	// printf("hash_join: finish hash_join\n");
	if(cudaThreadSynchronize() != cudaSuccess) {
		fprintf(stderr, "kernel failed at hash_join\n");
		ret = -1;
		goto failed;
	}

	// setting return values, if required
	if(h_res) {
		// restmp = (record_t *)malloc(n_results * sizeof(record_t));
		// if(!restmp) {
		// 	fprintf(stderr, "malloc failed for h_res\n");
		// 	ret = -1;
		// 	goto failed;
		// }
		res = cudaMallocHost((void**)&restmp, n_results * sizeof(record_t));
		if(res != cudaSuccess) {
			fprintf(stderr, "cudaMallocHost(&h_res) failed\n");
			ret = -1;
			goto failed;
		}

		// res = cudaMemcpy((void *)restmp, d_res, n_results * sizeof(record_t),
		//         cudaMemcpyDeviceToHost);
		// printf("hash_join: begin cudaMemcpyAsync\n");
		res = cudaMemcpyAsync((void *)restmp, d_res, n_results * sizeof(record_t),
		        cudaMemcpyDeviceToHost, Stream[StreamID]);
		// printf("hash_join: finish cudaMemcpyAsync\n");
		if (res != cudaSuccess) {
			printf("cudaMemcpyDtoH failed when getting join results\n");
			ret = -1;
			goto failed;
		}
		*h_res = restmp;
	}

	if(reslen) {
		*reslen = n_results;
	}

	goto finish;

failed:
	cudaFree(restmp);

finish:
	CUDA_FREE(d_s);
	CUDA_FREE(d_hist);
	CUDA_FREE(d_loc);
	CUDA_FREE(d_res);
	return ret;
}
