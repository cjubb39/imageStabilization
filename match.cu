#include <stdio.h>
#include <assert.h>

#include "error_handling.h"

#define PI 3.1415926
#define VECTOR_LENGTH 8

#define RANSAC_MIN_MATCHES 4
#define RANSAC_MAX_MATCHES 15
#define RANSAC_MAX_ATTEMPTS 100

#define RANSAC_ANGLE_EPSILON 0.5 /* radians */
#define RANSAC_ANGLE_THRESHOLD 0.75 /* percentage success */

#define RANSAC_TRANS_EPSILON 1
#define RANSAC_TRANS_THRESHOLD 0.90 /* percentage success */

#define RANSAC_EPSILON 0.90
#define RANSAC_THRESHOLD 0.75

#define RANSAC_NO_ANGLE_FOUND 123456789.123456678
#define RANSAC_NO_TRANS_FOUND 987654321.123456789

#define TRANSFORM_DEFAULT_VALUE 6893

__host__ void ransac_translation(double (*vectors)[VECTOR_LENGTH], int length, double *result){
	double trans_running_count_x = 0;
	double trans_running_count_y = 0;
	double trans_x[RANSAC_MAX_MATCHES];
	double trans_y[RANSAC_MAX_MATCHES];

	/* Use difference between base coordinates.  Eliminates need to rotate */
	for(int i = 0; i < length; ++i){
		trans_running_count_x += (trans_x[i] = vectors[i][4] - vectors[i][0]);
		trans_running_count_y += (trans_y[i] = vectors[i][5] - vectors[i][1]);
	}

	double trans_x_avg = trans_running_count_x / (double) length;
	double trans_y_avg = trans_running_count_y / (double) length;
	int threshold = RANSAC_TRANS_THRESHOLD*(length - 1) + 1;
	printf("x avg: %f; y avg: %f; threshold: %d; length: %d\n", trans_x_avg, trans_y_avg, threshold, length);

	int count_within_epsilon = 0;
	for(int i = 0; count_within_epsilon < threshold && i < length; ++i){
		if(abs(trans_x[i] - trans_x_avg) < RANSAC_TRANS_EPSILON && 
				abs(trans_y[i] - trans_y_avg) < RANSAC_TRANS_EPSILON){
			count_within_epsilon++;
		}
	}

	if (count_within_epsilon >= threshold){
		result[0] = trans_x_avg;
		result[1] = trans_y_avg;
	} else {
		result[0] = RANSAC_NO_TRANS_FOUND;
		result[1] = RANSAC_NO_TRANS_FOUND;
	}
}

__host__ double ransac_angle(double (*vectors)[VECTOR_LENGTH], int length){
	double toRet = RANSAC_NO_ANGLE_FOUND;

	double angle_running_count = 0;
	double theta[RANSAC_MAX_MATCHES];

	for(int i = 0; i < length; ++i){
		double *current = vectors[i];
		double tmp[VECTOR_LENGTH];

		/* make copy with v1, v2 starting at origin */
		tmp[0] = current[0] - current[0];
		tmp[1] = current[1] - current[1];
		tmp[2] = current[2] - current[0];
		tmp[3] = current[3] - current[1];
		tmp[4] = current[4] - current[4];
		tmp[5] = current[5] - current[5];
		tmp[6] = current[6] - current[4];
		tmp[7] = current[7] - current[5];

		/* length of vector */
		double l = sqrt((tmp[2] - tmp[0]) * (tmp[2] - tmp[0]) +
			(tmp[3] - tmp[1]) * (tmp[3] - tmp[1]));

		double d = sqrt((tmp[2] - tmp[6]) * (tmp[2] - tmp[6]) +
			(tmp[3] - tmp[7]) * (tmp[3] - tmp[7]));

		angle_running_count += (theta[i] = 2 * asin(d / (2*l)));
	}

	double theta_avg = angle_running_count / (double) length;
	int threshold = RANSAC_ANGLE_THRESHOLD*(length - 1) + 1;
	int count_within_epsilon = 0;
	for(int i = 0; count_within_epsilon < threshold && i < length; ++i){
		if (abs(theta[i] - theta_avg) < RANSAC_ANGLE_EPSILON)
			count_within_epsilon++;
	}

	if (count_within_epsilon >= threshold)
		toRet = theta_avg;

	return toRet;
}

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

/*	double theta[RANSAC_MAX_MATCHES];
	double t1[RANSAC_MAX_MATCHES];
	double t2[RANSAC_MAX_MATCHES];

	double	theta_running = 0,
					t1_running = 0,
					t2_running = 0;*/

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

/*		double	theta_m = atan(dy_m / dx_m),
						theta_s = atan(dy_s / dx_s),
						theta = theta_s - theta_m;*/

		double	t1_m = (cur[0] + cur[2]) / 2,
						t2_m = (cur[1] + cur[3]) / 2,
						t1_s = (cur[4] + cur[6]) / 2,
						t2_s = (cur[5] + cur[7]) / 2;

		double	theta_m = atan(dy_m / dx_m) + ((dx_m < 0) ? PI : 0),
						theta_s = atan(dy_s / dx_s) + ((dx_s < 0) ? PI : 0);
						/*theta = theta_s - theta_m*/

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

#ifdef DEBUG
printf("T_M: %f; T_S: %f; THETA: %f; cosT: %f; sinT: %f\n", 
	theta_m, theta_s, theta, cos(theta), sin(theta));
#endif
		/* rotate model to get intermediate vector used to get translation */
		/*double	intermediate_x = cos(theta)*t1_m - sin(theta)*t2_m,
						intermediate_y = sin(theta)*t1_m + cos(theta)*t2_m;*/
#ifdef DEBUG
printf("INTX: %f; INTY: %f\n", intermediate_x, intermediate_y);
#endif
		/*double	trans_x = intermediate_x - t1_s;
		double	trans_y = intermediate_y - t2_s;*/


		/* assemble transform matrix */
/*		rt_total[0][0] = cos(theta);
		rt_total[0][1] = -sin(theta);
		rt_total[0][2] = trans_x;
		rt_total[1][0] = sin(theta);
		rt_total[1][1] = cos(theta);
		rt_total[1][2] = trans_y;*/
		rt_total[2][0] = 0;
		rt_total[2][1] = 0;
		rt_total[2][2] = 1;
		#ifdef DEBUG
		for (int q = 0; q < 3; ++q){
			printf("%f %f %f\n", rt_total[q][0], rt_total[q][1], rt_total[q][2]);
		}
		#endif
/*		theta_running += (theta[i] = acos(rt_s[0][0]));
		t1_running += (t1[i] = rt_s[0][2]);
		t2_running += (t2[i] = rt_s[1][2]);*/
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

	/*double	theta_avg	= theta_running / length,
					t1_avg 		= t1_running / length,
					t2_avg 		= t2_running / length;

	printf("T: %f; 1: %f; 2: %f\n", theta_avg, t1_avg, t2_avg);
	int threshold_theta = RANSAC_ANGLE_THRESHOLD*(length - 1) + 1,
			threshold_t1		= RANSAC_TRANS_THRESHOLD*(length - 1) + 1,
			threshold_t2		= threshold_t1;
*/
	/* check thetas agree */
	/*int count_within_epsilon = 0;
	for(int i = 0; count_within_epsilon < threshold_theta && i < length; ++i){
		if (abs(theta[i] - theta_avg) < RANSAC_ANGLE_EPSILON)
			count_within_epsilon++;
	}

	if (count_within_epsilon < threshold_theta)
		return;
	*/
	/* check t1 agree */
	/*count_within_epsilon = 0;
	for(int i = 0; count_within_epsilon < threshold_t1 && i < length; ++i){
		if (abs(t1[i] - t1_avg) < RANSAC_TRANS_EPSILON)
			count_within_epsilon++;
	}

	if (count_within_epsilon < threshold_t1)
		return;
	*/
	/* check t2 agree */
	/*count_within_epsilon = 0;
	for(int i = 0; count_within_epsilon < threshold_t2 && i < length; ++i){
		if (abs(t2[i] - t2_avg) < RANSAC_TRANS_EPSILON)
			count_within_epsilon++;
	}

	if (count_within_epsilon < threshold_t2)
		return;
*/

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
		if (*tracker == char1){
			index1_im2 = i;
		} else if (*tracker == char2){
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
 *		model (im1) -> scene (im2)
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
				/*theta = ransac_angle(vectors, matches);
				if (theta == RANSAC_NO_ANGLE_FOUND){
					continue;
				}
				ransac_translation(vectors, matches, translation);
				if (translation[0] == RANSAC_NO_TRANS_FOUND){
					continue;
				}*/

				ransac_full(vectors, matches, matrix);

				if (matrix[0] == TRANSFORM_DEFAULT_VALUE){
					continue;
				}
				/* passed both theta / translation tests */
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
}

int main(void){
	srand(time(NULL));
	double *matrix = (double *) malloc(sizeof(double) * 9);

	for(int i = 0; i < 9; ++i)
		matrix[i] = TRANSFORM_DEFAULT_VALUE;

	/*double vec1_1[3] = {1,1,1};
	double vec1_2[3] = {1,1,1};

	double vec2_1[3] = {2,2,2};
	double vec2_2[3] = {2,2,2};

	double vec3_1[3] = {3,3,3};
	double vec3_2[3] = {3,3,3};

	double vec4_1[3] = {4,4,4};
	double vec4_2[3] = {4,4,4};
	
	double vec5_1[3] = {5,5,5};
	double vec5_2[3] = {5,5,5};

	double vec6_1[3] = {6,6,6};
	double vec6_2[3] = {6,6,6};

	double vec7_1[3] = {7,7,7};
	double vec7_2[3] = {7,7,7};

	double vec8_1[3] = {8,8,8};
	double vec8_2[3] = {8,8,8};

	double **im1 = (double **) malloc(sizeof(double*) * 8);
	im1[0] = vec1_1;
	im1[1] = vec2_1;
	im1[2] = vec3_1;
	im1[3] = vec4_1;
	im1[4] = vec5_1;
	im1[5] = vec6_1;
	im1[6] = vec7_1;
	im1[7] = vec8_1;

	double **im2 = (double **) malloc(sizeof(double*) * 8);
	im2[0] = vec1_2;
	im2[1] = vec2_2;
	im2[2] = vec3_2;
	im2[3] = vec4_2;
	im2[4] = vec5_2;
	im2[5] = vec6_2;
	im2[6] = vec7_2;
	im2[7] = vec8_2;*/

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
