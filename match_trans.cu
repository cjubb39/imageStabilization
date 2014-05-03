#include <cuda.h>
#include "error_handling.h"
#include "match.h"
#include "transform.h"
#include "mainSift.h"
#include "im1.h"

__host__ int main(int argc, char **argv)
{
	if (argc < 3){
		fprintf(stderr, "run as %s <file_prefix> <count>\n", argv[0]);
		exit(1);
	}

	srand(time(NULL));

	int num_img = atoi(argv[2]);
	printf("NUM IMAGES: %d\n", num_img);
	printf("Identifying Features...\n");

	double *img1_featurelist;
	double *img2_featurelist;
	int img1_feat_length;
	int img2_feat_length;

	float *transforms_f = (float *) malloc(sizeof(float) * 9 * num_img);
	for(int i = 0; i < 9 * num_img; ++i)
	{
		transforms_f[i] = 0;
	}
	transforms_f[0] = 1;
	transforms_f[4] = 1;
	transforms_f[8] = 1;

	for(int i = 9 ; i < 9 * num_img; ++i){
		transforms_f[i] = TRANSFORM_DEFAULT_VALUE;
	}

	char name_buffer1[1024];
	char name_buffer2[1024];

	for(int i = 0; i < num_img - 1; ++i){
		sprintf(name_buffer1, "%s%02d.exr", argv[1], i + 1);
		sprintf(name_buffer2, "%s%02d.exr", argv[1], i + 2);

		//printf("img1: %s\n", name_buffer1);
		//printf("img2: %s\n", name_buffer2);
		
		sift_images(name_buffer1, name_buffer2, &img1_featurelist, \
			&img1_feat_length, &img2_featurelist, &img2_feat_length, \
			&transforms_f[(9*(i + 1))]);
	}

	//free(img1_featurelist);
	//free(img2_featurelist);
//
//	int num_img = 4;
//
//	double *transforms = (double *) malloc(sizeof(double) * 9 * num_img);
//
//	for(int i = 0; i < 9 * num_img; ++i)
//	{
//		transforms[i] = 0;
//	}
//	transforms[0] = 1;
//	transforms[4] = 1;
//	transforms[8] = 1;
//
//	double im1[27] = {0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8};
//	double im2[27] = {0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8};
//	double im3[27] = {0,0,0,-1,-1,1,-2,-2,2,-3,-3,3,-4,-4,4,-5,-5,5,-6,-6,6,-7,-7,7,-8,-8,8};
//	double im4[27] = {50,20,0,51,21,1,52,22,2,53,23,3,54,24,4,55,25,5,56,26,6,57,27,7,58,28,8};
//
//	runtest(im1, im2, &transforms[9]);
//	runtest(im2, im3, &transforms[18]);
//	runtest(im3, im4, &transforms[27]);
//
	
/*	for (int i = 0; i < 9 * num_img; i += 3){
		printf("%f %f %f\n", transforms_f[i], transforms_f[i + 1], transforms_f[i+2]);
		if((i % 9) == 6)
			printf("\n");
	}*/


	/* convert transforms from above */
	double *transforms = (double *) malloc(sizeof(double) * 9 * num_img);

	for (int i=0; i < 9 * num_img; ++i){
		transforms[i] = transforms_f[i];
	}
	
	comb_trans(transforms, num_img);

	/* get height and width information */
	int width, height;
	float *input;
	sprintf(name_buffer1, "%s%02d.exr", argv[1], 1);
	readOpenEXRFile(name_buffer1, &input, width, height);


	/* get xtrans, ytrans, dwidth, and dheight */
	int xtrans;
	int ytrans;
	int dwidth;
	int dheight;

	find_dest_multi(transforms, num_img, width, height, &xtrans, &ytrans, 
		&dwidth, &dheight);


	/* apply transform and output */
	float *output = (float *) malloc(sizeof(float) * dwidth * dheight * 3);
	char out_name [30];
	for (int i = 0; i < num_img; i++)
	{
		for (int j = 0; j < dwidth*dheight*3; j += 3)
		{
			output[j] = 0;
			output[j + 1] = 1;
			output[j + 2] = 0;
		}
		
		sprintf(name_buffer1, "%s%02d.exr", argv[1], i + 1);
		readOpenEXRFile(name_buffer1, &input, width, height);

		apply_transform(input, output, &transforms[9*i], width, height, xtrans, ytrans, dwidth, dheight);

		sprintf(out_name, "t_%s%02d.exr", argv[1], i + 1);
		writeOpenEXRFile(out_name, output, dwidth, dheight);
	}


	/* clean up */
	free(transforms);
	free(transforms_f);
	cudaFree(input);
	free(output);

	return 1;
}
