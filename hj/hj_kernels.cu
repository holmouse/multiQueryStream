#include "hj.h"

// this is the version without avoiding memory bank conflicts
// TODO: change the work distribution scheme: use deterministic
// number of thread blocks to process any long array
__global__ void prefix_sum(int *d_dest, int *d_sumbuf, int *d_src,
        int nr_chunks, int len)
{
	extern __shared__ float temp[];
	int tid = threadIdx.x;

	// for each chunk handled by this thread block
	for(int i = blockIdx.x; i < nr_chunks; i += gridDim.x) {
		int *idata = d_src + 2 * blockDim.x * i;
		int *odata = d_dest + 2 * blockDim.x * i;
		int offset = 1;

		// how many elements to be processed by this thread block
		int n = 2 * blockDim.x * (i + 1);
		if(n > len) {
			n = len % (2 * blockDim.x);
		}
		else {
			n = 2 * blockDim.x;
		}

		// load elements to shared memory
		for(int j = tid; j < n; j += blockDim.x) {
			temp[j] = idata[j];
		}
		// zero the rest elements in shared memory, if any
		// we zero elements from right end to left until n; this strange
		// design is to ensure the last element is zeroed by the last thread
		// in the block, if it is required, in order to prevent a rare RAW
		// antidependency from happening: the last thread reading the last
		// element of the shared memory in previous loop VS. some thread
		// zeroing this element in the next loop
		for(int j = blockDim.x + tid; j >= n; j -= blockDim.x) {
			temp[j] = 0;
		}

		// the up-sweep phase
		for(int d = blockDim.x; d > 0; d >>= 1) {
			__syncthreads();
			if (tid < d) {
				int ai = offset*(2*tid+1)-1;
				int bi = offset*(2*tid+2)-1;
				temp[bi] += temp[ai];
			}
			offset *= 2;
		}

		// clear the last element
		if(tid == 0) {
			temp[2 * blockDim.x - 1] = 0;
		}

		// the down-sweep phase
		for (int d = 1; d < 2 * blockDim.x; d *= 2) {
			offset >>= 1;
			__syncthreads();

			if(tid < d) {
				int ai = offset*(2*tid+1)-1;
				int bi = offset*(2*tid+2)-1;
				int t = temp[ai];
				temp[ai] = temp[bi];
				temp[bi] += t;
			}
		}
		__syncthreads();

		// write back result
		for(int j = threadIdx.x; j < n; j += blockDim.x) {
			odata[j] = temp[j];
		}

		// the last thread writes the sumbuf
		if(tid == blockDim.x - 1) {
			d_sumbuf[i] = temp[2 * blockDim.x - 1];
		}
	} // for each chunk
}

__global__ void prefix_sum_update(int *d_dest, int *d_sumbuf, int nr_chunks,
        int len)
{
	// for each chunk handled by this thread block
	for(int i = blockIdx.x; i < nr_chunks; i += gridDim.x) {
		int *odata = d_dest + 2 * blockDim.x * i;
		int delta = d_sumbuf[i];

		// how many elements to be processed by this thread block
		int n = 2 * blockDim.x * (i + 1);
		if(n > len) {
			n = len % (2 * blockDim.x);
		}
		else {
			n = 2 * blockDim.x;
		}

		// update the sums
		for(int j = threadIdx.x; j < n; j += blockDim.x) {
			odata[j] += delta;
		}
	}
}

// the layout of d_hist: d_hist[n_buckets][gridDim.x * blockDim.x]
// the version without using shared memory
__global__ void hash_build_hist(int *d_hist, record_t *d_r, int rlen,
        int n_buckets)
{
	const int nr_threads_tot = gridDim.x * blockDim.x;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int offset1 = offset;

	for(int i_bucket = 0; i_bucket < n_buckets; ++i_bucket) {
		d_hist[i_bucket * nr_threads_tot + offset] = 0;
	}

	while(offset1 < rlen) {
	    // n_buckets has to be 2^n
		int i_bucket = HASH(d_r[offset1].y) & (n_buckets - 1);
		++d_hist[i_bucket * nr_threads_tot + offset];
		offset1 += nr_threads_tot;
	}
}

// after hash table is built, records in the ith bucket are stored
// in d_rec[d_idx[i], d_idx[i+1])
__global__ void hash_build(record_t* d_rec, int* d_idx, record_t* d_r,
        int rlen, int* d_loc, int n_buckets)
{
	int nr_threads_tot = gridDim.x * blockDim.x;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int offset1 = offset;

	// the first warp of the first thread block writes d_idx
	if(offset < 32) {
		for(int i_bucket = threadIdx.x; i_bucket < n_buckets; i_bucket += 32) {
			d_idx[i_bucket] = d_loc[i_bucket * nr_threads_tot];
		}
		if(threadIdx.x == 0) {
			d_idx[n_buckets] = rlen;
		}
	}

	while(offset1 < rlen) {
		record_t r = d_r[offset1];
		int i_loc = (HASH(r.y) & (n_buckets - 1)) * nr_threads_tot + offset;
		d_rec[d_loc[i_loc]++] = r;
		offset1 += nr_threads_tot;
	}
}

// hash_join_hist and hash_join distribute work like this:
// number of thread blocks and number of threads per thread block
// are predefined and deterministic;
// records in s are evenly allocated to all warps;
// for each record allocated to each warp, search in the specific
// bucket is done in parallel by all threads within the warp, and
// each thread outputs result on its own
__global__ void hash_join_hist(int *d_hist, record_t *d_hash_rec,
        int *d_hash_idx, int n_buckets, record_t *d_s, int slen)
{
	int nr_warps_tot = (blockDim.x / 32) * gridDim.x;
	int count = 0;

	// for each record allocated to this thread warp
	for(int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; i < slen;
	        i += nr_warps_tot) {
		record_t s = d_s[i];
		int i_bucket = HASH(s.y) & (n_buckets - 1);
		int n = d_hash_idx[i_bucket + 1] - d_hash_idx[i_bucket];
		record_t *bucket_begin = d_hash_rec + d_hash_idx[i_bucket];

		// for each hashed record in the bucket
		for(int j = threadIdx.x % 32; j < n; j += 32) {
			if(bucket_begin[j].y == s.y) {
				count++;
			}
		}
	}

	// write how many join results generated by this thread
	d_hist[blockIdx.x * blockDim.x + threadIdx.x] = count;
}

__global__ void hash_join(record_t *d_res, int *d_loc, record_t *d_hash_rec,
        int *d_hash_idx, int n_buckets, record_t *d_s, int slen)
{
	int nr_warps_tot = (blockDim.x / 32) * gridDim.x;
	int offset = d_loc[blockIdx.x * blockDim.x + threadIdx.x];
	record_t res;

	// for each record allocated to this thread warp
	for(int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; i < slen;
	        i += nr_warps_tot) {
		record_t s = d_s[i];
		int i_bucket = HASH(s.y) & (n_buckets - 1);
		int n = d_hash_idx[i_bucket + 1] - d_hash_idx[i_bucket];
		record_t *bucket_begin = d_hash_rec + d_hash_idx[i_bucket];

		// for each hashed record in the bucket
		for(int j = threadIdx.x % 32; j < n; j += 32) {
			if(bucket_begin[j].y == s.y) {
				res.x = bucket_begin[j].x;
				res.y = s.x;
				d_res[offset++] = res;
			}
		}
	}
}
