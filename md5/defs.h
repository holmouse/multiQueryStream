#ifndef _DEFS_H_
#define _DEFS_H_

#define BYTES_INPUT	(64L * 1024L * 1024L)
#define MD5_LEN			16

#define DIFF_TIME(t1, t2) ( \
    ((t2).tv_sec + ((double)(t2).tv_usec)/1000000.0) - \
    ((t1).tv_sec + ((double)(t1).tv_usec)/1000000.0) \
)

#endif
