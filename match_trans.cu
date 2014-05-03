#include <cuda.h>
#include "error_handling.h"
#include "match.h"
#include "transform.h"
#include "mainSift.h"
#include "im1.h"

__host__ int main(int argc, char **argv)
{
	if (argc < 3){
		fprintf(stderr, "run as %s <image 1> <image 2>\n", argv[0]);
		exit(1);
	}

	srand(time(NULL));

	int num_img = 2;

	double *img1_featurelist;
	double *img2_featurelist;
	int img1_feat_length;
	int img2_feat_length;

	float *transforms_f = (float *) malloc(sizeof(float) * 18);
		for(int i = 0; i < 9 * num_img; ++i)
	{
		transforms_f[i] = 0;
	}
	transforms_f[0] = 1;
	transforms_f[4] = 1;
	transforms_f[8] = 1;

	for(int i = 9 ; i < 18; ++i){
		transforms_f[i] = TRANSFORM_DEFAULT_VALUE;
	}

	sift_images(argv[1], argv[2], &img1_featurelist, &img1_feat_length, &img2_featurelist, &img2_feat_length, &transforms_f[9]);

	//fprintf(stderr, "IM1 LIST:\n");
	for (int i = 0; i < img1_feat_length; i+=3)
	{
		img1_featurelist[i] -= 1280/2;
		img1_featurelist[i + 1] -= 960/2;
	}

	//fprintf(stderr, "\nIM2 LIST:\n");
	for (int i = 0; i < img2_feat_length; i+=3)
	{
		img2_featurelist[i] -= 1280/2;
		img2_featurelist[i + 1] -= 960/2;
	}


	for(int i = 0; i < 18; ++i){
		printf("%f ", transforms_f[i]);
		if((i+1) % 3 == 0)
			printf("\n");

		if((i+1) %9 == 0)
			printf("\n");
	}

/*	transforms_f[9] = 0.997953;
	transforms_f[10] = 0.003084;
	transforms_f[11] = -1.863;

	transforms_f[12] = 0.006153;
	transforms_f[13] = 0.996806;
	transforms_f[14] = 10.044221;
	
	transforms_f[15] = 0;//-0.000020;
	transforms_f[16] = 0;//0.000017;
	transforms_f[17] = 1;*/

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
//	comb_trans(transforms, num_img);

	int width, height;
	float *input;
	readOpenEXRFile(argv[1], &input, width, height);

	int xtrans;
	int ytrans;
	int dwidth;
	int dheight;

	double *transforms = (double *) malloc(sizeof(double) * 18);

	for (int i=0; i < 18; ++i){
		transforms[i] = transforms_f[i];
	}

	find_dest_multi(transforms, num_img, width, height, &xtrans, &ytrans, 
		&dwidth, &dheight);

	float *output = (float *) malloc(sizeof(float) * dwidth * dheight * 3);
	for (int i = 0; i < dwidth*dheight*3; i += 3)
	{
		output[i] = 0;
		output[i + 1] = 1;
		output[i + 2] = 0;
	}

	char name [30];
	for (int i = 0; i < num_img; i++)
	{
		readOpenEXRFile(argv[i+1], &input, width, height);
		apply_transform(input, output, &transforms[9*i], width, height, xtrans, ytrans, dwidth, dheight);
		sprintf(name, "t_%s", argv[i + 1]);
		writeOpenEXRFile(name, output, dwidth, dheight);
	}

	free(transforms);
	free(transforms_f);
	cudaFree(input);
	free(output);

	return 1;
}
