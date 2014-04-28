#include <stdio.h>
#include <assert.h>

#include "error_handling.h"

#define PI 3.1415926
#define VECTOR_LENGTH 8

#define RANSAC_MIN_MATCHES 4
#define RANSAC_MAX_MATCHES 15
#define RANSAC_MAX_ATTEMPTS 100

#define RANSAC_EPSILON 0.90
#define RANSAC_THRESHOLD 0.75

#define RANSAC_CHARACTERISTIC_THRESHOLD 1

#define TRANSFORM_DEFAULT_VALUE 6893

__host__ void transform(double *pt_before, double (*matrix)[3], double *pt_after){
	for (int i = 0; i < 3; ++i){
		pt_after[i] =
			matrix[i][0] * pt_before[0] +
			matrix[i][1] * pt_before[1] +
			matrix[i][2] * pt_before[2];
	}
}

/* first four in a single *vector as model; second four as scene */
__host__ void ransac_full(double(*vectors)[VECTOR_LENGTH], int length, double *result){
	assert(length <= RANSAC_MAX_MATCHES);

	/* get local copy of vector pointers */
	double **local_vectors = (double **) malloc(sizeof(double*) * length);

	for(int i = 0; i < length; ++i){
		local_vectors[i] = vectors[i];
	}

	/* randomly select vector to use as reference */
	int tmp_move = rand() % length;
	
	#ifdef DEBUG
	printf("SWAP: %d\n", tmp_move);
	#endif

	double *tmp = local_vectors[0];
	local_vectors[0] = local_vectors[tmp_move];
	local_vectors[tmp_move] = tmp;

	/* calculate vector between random features */
	double rt_total[3][3];

	{
		double *cur = local_vectors[0];

		double	dx_m = cur[0] - cur[2], 
						dy_m = cur[1] - cur[3],
						dx_s = cur[4] - cur[6],
						dy_s = cur[5] - cur[7];

		double	t1_m = (cur[0] + cur[2]) / 2,
						t2_m = (cur[1] + cur[3]) / 2,
						t1_s = (cur[4] + cur[6]) / 2,
						t2_s = (cur[5] + cur[7]) / 2;

		double	theta_m = atan(dy_m / dx_m) + ((dx_m < 0) ? PI : 0),
						theta_s = atan(dy_s / dx_s) + ((dx_s < 0) ? PI : 0);

		double	rt_m_inverse[3][3], rt_s[3][3];

		/* assemble matricies representing rotation / translation of model/scene */
		rt_m_inverse[0][0] = cos(theta_m);
		rt_m_inverse[0][1] = sin(theta_m);
		rt_m_inverse[0][2] = -t1_m*cos(theta_m) - t2_m*sin(theta_m);
		rt_m_inverse[1][0] = -sin(theta_m);
		rt_m_inverse[1][1] = cos(theta_m);
		rt_m_inverse[1][2] = t1_m*sin(theta_m) - t2_m*cos(theta_m);

		rt_s[0][0] = cos(theta_s);
		rt_s[0][1] = -sin(theta_s);
		rt_s[0][2] = t1_s;
		rt_s[1][0] = sin(theta_s);
		rt_s[1][1] = cos(theta_s);
		rt_s[1][2] = t2_s;

		rt_m_inverse[2][0] = rt_s[2][0] = 0;
		rt_m_inverse[2][1] = rt_s[2][1] = 0;
		rt_m_inverse[2][2] = rt_s[2][2] = 1;
	
		/* find overall rotation matrix = rt_s*rt_m_inverse */
		for (int j = 0; j < (3 - 1); ++j){
			for (int k = 0; k < 3; ++k){
				rt_total[j][k] = 
					rt_s[j][0] * rt_m_inverse[0][k] +
					rt_s[j][1] * rt_m_inverse[1][k] +
					rt_s[j][2] * rt_m_inverse[2][k];
			}
		}
		rt_total[2][0] = 0;
		rt_total[2][1] = 0;
		rt_total[2][2] = 1;
		
		#ifdef DEBUG
		for (int q = 0; q < 3; ++q){
			printf("%f %f %f\n", rt_total[q][0], rt_total[q][1], rt_total[q][2]);
		}
		#endif
	}


	/* check if matrix works for other points */
	
	/* we include the point for which the matrix was calculated */
	int count_within_epsilon = 1;
	int threshold = RANSAC_THRESHOLD*(length - 1) + 1;
	
	for (int i = 1; i < length; ++i){
		double *cur=local_vectors[i];

		#ifdef DEBUG
		for(int q = 0; q < 8; ++q){
		printf("%f ", cur[q]);
		}
		printf("\n");
		#endif

		double pt_before[3];
		double pt_after[3];

		pt_before[0] = cur[0];
		pt_before[1] = cur[1];
		pt_before[2] = 1;
		transform(pt_before, rt_total, pt_after);

		#ifdef DEBUG
		printf("ptBx: %f; expected: %f;; ptBy: %f; expected %f\n",
		pt_before[0], cur[0], pt_before[1], cur[1]);
		printf("ptAx: %f; expected: %f;; ptAy: %f; expected %f\n\n",
		pt_after[0], cur[4], pt_after[1], cur[5]);
		#endif

		if(abs(pt_after[0] - cur[4]) > RANSAC_THRESHOLD ||
				abs(pt_after[1] - cur[5]) > RANSAC_THRESHOLD){
			continue;
		}

		/* check second point in vector */
		pt_before[0] = cur[2];
		pt_before[1] = cur[3];
		pt_before[2] = 1;
		transform(pt_before, rt_total, pt_after);

		if(abs(pt_after[0] - cur[6]) > RANSAC_THRESHOLD ||
				abs(pt_after[1] - cur[7]) > RANSAC_THRESHOLD){
			continue;
		}

		++count_within_epsilon;
	}

	if (count_within_epsilon < threshold)
		return;


	/* SUCCESS! */
	for (int i = 0; i < 9; i++){
		result[i] = rt_total[i/3][i%3];
	}

}

__host__ int get_random_vector_pair_diff(double *im1, int im1_len, 
	double *im2, int im2_len, double result[VECTOR_LENGTH]){

	int index1 = rand() % im1_len; 
	int index2 = rand() % im1_len;
	int index1_im2 = -1;
	int index2_im2 = -1;

	/* get characteristic value from random index */
	double char1 = im1[3*index1 + 2];
	double char2 = im1[3*index2 + 2];
	
	int i = 0;
	double *tracker;
	for (i = 0, tracker = (im2 + 2); i < im2_len; tracker += 3, ++i){
		if (abs(*tracker - char1) < RANSAC_CHARACTERISTIC_THRESHOLD){
			index1_im2 = i;
		} else if (abs(*tracker - char2) < RANSAC_CHARACTERISTIC_THRESHOLD){
			index2_im2 = i;
		}
	}

	if (index1_im2 == -1 || index2_im2 == -1)
		return -1; /* try again */


	/* at this point, we have two points with the same two random salient points */
	result[0] = im1[index1*3]; /* 1.1.x */
	result[1] = im1[index1*3 + 1]; /* 1.1.y */
	result[2] = im1[index2*3]; /* 1.2.x */
	result[3] = im1[index2*3 + 1]; /* 1.2.y */
	result[4] = im2[index1_im2*3]; /* 2.1.x */
	result[5] = im2[index1_im2*3 + 1]; /* 2.1.y */
	result[6] = im2[index2_im2*3]; /* 2.2.x */
	result[7] = im2[index2_im2*3 + 1]; /* 2.2.y */

	return 0;
}


/*
 *	Takes in list of "salient features": (x, y, characteristic) for two images
 *	Lists are single dimensional vector in row-major order (all doubles)
 *	
 *	Returns representation of translation / rotation necessary to match images
 *		scene (im2) -> model (im1)
 *	
 */
__host__ void match_images(double *im1, int im1_len, 
		double *im2, int im2_len, double *matrix){

	/* number of valid vectors between salient points so far */
	int matches = 0;

	/*	hold x,y of translation/rotation vector
	 *	[0]: 1.1.x
	 *	[1]: 1.1.y
	 *	[2]: 1.2.x
	 *	[3]: 1.2.y 
	 *	[4]: 2.1.x
	 *	[5]: 2.1.y
	 *	[6]: 2.2.x
	 *	[7]: 2.2.y
	 */
	double vectors[RANSAC_MAX_MATCHES][VECTOR_LENGTH];

	//int max = (im1_len > im2_len) ? im1_len : im2_len;
	//int min = im1_len ^ im2_len ^ max;

	/* get initial data sampling */
	for (int i = 0; i < RANSAC_MAX_ATTEMPTS; ++i) {
		matches = 0;
		while(matches < RANSAC_MAX_MATCHES && matches < im1_len){
			double points[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
			if (get_random_vector_pair_diff(im1, im1_len, im2, im2_len, points) == 0){
				vectors[matches][0] = points[0];
				vectors[matches][1] = points[1];
				vectors[matches][2] = points[2];
				vectors[matches][3] = points[3];
				vectors[matches][4] = points[4];
				vectors[matches][5] = points[5];
				vectors[matches][6] = points[6];
				vectors[matches][7] = points[7];
				++matches;
			} else {
				continue; /* no match; try again */
			}

			if (matches >= RANSAC_MIN_MATCHES) {
				ransac_full(vectors, matches, matrix);

				if (matrix[0] == TRANSFORM_DEFAULT_VALUE){
					continue;
				}
				/* passed both theta / translation tests */

				/* adjust so we are returning scene -> model matrix */
				matrix[1] *= -1;
				matrix[2] *= -1;
				matrix[3] *= -1;
				matrix[5] *= -1;

				return;
			}
		}
	}

	return; /* no match, quit */
}

void runtest(double *im1, double *im2, double *matrix){
	for(int i = 0; i < 9; ++i)
		matrix[i] = TRANSFORM_DEFAULT_VALUE;
	
	match_images(im1, 9, im2, 9, matrix);

	for (int i = 0; i < 9; ++i){
		printf("%f ", matrix[i]);
		if(i%3 == 2)
			printf("\n");
	}
	printf("\n");
}

int main(void){
	srand(time(NULL));
	double *matrix = (double *) malloc(sizeof(double) * 9);

	for(int i = 0; i < 9; ++i)
		matrix[i] = TRANSFORM_DEFAULT_VALUE;

	double im1[27] = {0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8};
	
	double im2[27] = {0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8};
	runtest(im1, im2, matrix);

	double im2_2[27] = {0,0,0,-1,-1,1,-2,-2,2,-3,-3,3,-4,-4,4,-5,-5,5,-6,-6,6,-7,-7,7,-8,-8,8};
	runtest(im1, im2_2, matrix);

	double im2_3[27] = {5,2,0,6,3,1,7,4,2,8,5,3,9,6,4,10,7,5,11,8,6,12,9,7,13,10,8};
	runtest(im1, im2_3, matrix);

	double im2_4[27] = {5,22,0,6,23,1,7,24,2,8,5,3,9,6,4,10,7.2,5,11,8,6,12,9,7,13,10,8};
	runtest(im1, im2_4, matrix);

	double im2_5[27] = {5,2.1,0,6,3.1,1,7.1,4,2,8,5,3,9,6,4,10.1,7,5,11,8,6,12,9,7,13.1,10.1,8};
	runtest(im1, im2_5, matrix);

	free(matrix);
	return 0;
}
