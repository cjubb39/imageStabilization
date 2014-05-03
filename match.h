#ifndef MATCH_H
#define MATCH_H

#define PI 3.1415926
#define VECTOR_LENGTH 8

#define RANSAC_MIN_MATCHES 5
#define RANSAC_MAX_MATCHES 250
#define RANSAC_MAX_ATTEMPTS 10000

#define RANSAC_EPSILON 100
#define RANSAC_THRESHOLD 0.25

#define RANSAC_CHARACTERISTIC_THRESHOLD 5

#define TRANSFORM_DEFAULT_VALUE 6893

//Matching functions
__host__ void transform(double *pt_before, double (*matrix)[3], double *pt_after);
__host__ void ransac_full(double(*vectors)[VECTOR_LENGTH], int length, double *result);
__host__ int get_random_vector_pair_diff(double *im1, int im1_len,
	double *im2, int im2_len, double result[VECTOR_LENGTH]);
void match_images(double *, int, double *, int, double *);
void runtest(double *im1, double *im2, double *matrix);

#endif
