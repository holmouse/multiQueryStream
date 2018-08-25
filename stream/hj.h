#ifndef HJ_H
#define HJ_H

#include <sys/time.h>
#include <cuda.h>

#define TVAL(t)             ((t).tv_sec * 1000.0 + (t).tv_usec / 1000.0)
#define TIME_DIFF(t1, t2)   (TVAL(t2) - TVAL(t1))

#define FREE(p)			do { if(p) free(p); } while(0)
#define CUDA_FREE(p)	do { if(p) cudaFree(p); } while(0)

#define HASH(v)	((unsigned int)( (v >> 7) ^ (v >> 13) ^ (v >>21) ^ (v) ))

typedef struct int2_struct
{
	int x;
	int y;
} record_t;

typedef struct hash_table_struct
{
    record_t *d_rec;
	int n_records;
	int *d_idx;
	int n_buckets;
} hash_table_t;

int build_hash_table(hash_table_t *ht, record_t *h_r, int rlen, cudaStream_t* Stream, int StreamID);
void free_hash_table(hash_table_t *ht);
int hash_join(record_t **h_res, int *reslen, hash_table_t *ht_r,
        record_t *h_s, int slen, cudaStream_t* Stream, int StreamID);
int scan(int *d_dest, int *d_src, int len, cudaStream_t* Stream, int StreamID);

#endif
