// Memory object sizes:
// 1. hash table build: 2 * 8 * RLEN + 2 * 32 * 1024 * RBUCKETS
// 2. after hash_build before hash_join: 8 * RLEN
// 3. each hash_join: 8 * S_CHUNK_LEN + 8 * RLEN + 8 * n_results
#include <stdio.h>
#include <stdlib.h>

#include "hj.h"

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

int read_r(record_t **r, int *rlen)
{
	size_t size_r = sizeof(record_t) * RLEN;

	record_t *r_tmp = (record_t *)malloc(size_r);
	if(!r_tmp) {
		fprintf(stderr, "malloc failed for R\n");
		return -1;
	}

	unsigned int seed = RKEY_SEED;
	for(int i = 0; i < RLEN; i++) {
		r_tmp[i].y = rand_r(&seed) % RKEY_MAX;
		r_tmp[i].x = i;
	}

	*r = r_tmp;
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
	record_t *h_r, *h_s;
	hash_table_t ht_r;
	int rlen, slen;
	struct timeval t1, t2, t_start, t_end;

	gettimeofday(&t_start, NULL);

	printf("Time of starting hj: %lf \n", TVAL(t_start));

	// read r and build hash table
	gettimeofday(&t1, NULL);

	if(read_r(&h_r, &rlen)) {
		fprintf(stderr, "failed to read r\n");
		return -1;
	}

	gettimeofday(&t2, NULL);
	printf("Time on reading R: %lf ms\n", TIME_DIFF(t1, t2));

	gettimeofday(&t1, NULL);

	ht_r.n_buckets = RBUCKETS;
	if(build_hash_table(&ht_r, h_r, rlen)) {
		fprintf(stderr, "failed to build hash table for R\n");
		free(h_r);
		return -1;
	}

	gettimeofday(&t2, NULL);
	printf("Time on building hash table for R: %lf ms\n", TIME_DIFF(t1, t2));

	FREE(h_r);	// table R on the host is not needed any more

	// for each chunk of s, join with r
	h_s = (record_t *)malloc(sizeof(record_t) * S_CHUNK_LEN);
	if(!h_s) {
		fprintf(stderr, "malloc failed for s\n");
		free_hash_table(&ht_r);
		return -1;
	}

	// The number of result records joined per chunk is approximately:
	// RLEN * S_CHUNK_LEN / max(RKEY_MAX, SKEY_MAX)
	double t_read_s = 0.0, t_hash_join = 0.0;
	for(int i = 0; i < S_CHUNK_CNT; i++) {
		gettimeofday(&t1, NULL);
		slen = read_s(h_s, S_CHUNK_LEN, 0);	
		gettimeofday(&t2, NULL);
		t_read_s += TIME_DIFF(t1, t2);

		// join with r
		gettimeofday(&t1, NULL);
		if(slen > 0) {
			if(hash_join(NULL, NULL, &ht_r, h_s, slen)) {
				fprintf(stderr, "hash join failed for the %dth chunk of S\n",
				        i);
				break;
			}
		}
		else {
			fprintf(stderr, "failed to read s\n");
			break;
		}
		gettimeofday(&t2, NULL);
		t_hash_join += TIME_DIFF(t1, t2);
	}

	printf(	"Time on joining S with R: %lf ms\n"
			"\tTime on reading S: %lf ms ( %lf ms per read )\n"
			"\tTime on hash join: %lf ms ( %lf ms per join )\n",
			t_read_s + t_hash_join,
			t_read_s, t_read_s / S_CHUNK_CNT,
			t_hash_join, t_hash_join / S_CHUNK_CNT);
	free_hash_table(&ht_r);
	FREE(h_s);

	gettimeofday(&t_end, NULL);
	printf("Total time taken: %lf ms\n", TIME_DIFF(t_start, t_end));
	printf("Time of ending hj: %lf \n", TVAL(t_end));
	return 0;
}
