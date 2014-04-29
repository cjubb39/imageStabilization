#include <cuda.h>
#include "error_handling.h"
#include "match.h"
#include "transform.h"
#include "im1.h"

__host__ int main(int argc, char **argv)
{
	srand(time(NULL));

	int num_img = 4;

	double *transforms = (double *) malloc(sizeof(double) * 9 * num_img);

	for(int i = 0; i < 9 * num_img; ++i)
	{
		transforms[i] = 0;
	}
	transforms[0] = 1;
	transforms[4] = 1;
	transforms[8] = 1;

	double im1[27] = {0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8};
	double im2[27] = {0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8};
	double im3[27] = {0,0,0,-1,-1,1,-2,-2,2,-3,-3,3,-4,-4,4,-5,-5,5,-6,-6,6,-7,-7,7,-8,-8,8};
	double im4[27] = {50,20,0,51,21,1,52,22,2,53,23,3,54,24,4,55,25,5,56,26,6,57,27,7,58,28,8};

	runtest(im1, im2, &transforms[9]);
	runtest(im2, im3, &transforms[18]);
	runtest(im3, im4, &transforms[27]);

	comb_trans(transforms, num_img);

	int width, height;
	float *input;
	readOpenEXRFile(argv[1], &input, width, height);

	int xtrans;
	int ytrans;
	int dwidth;
	int dheight;

	find_dest_multi(transforms, num_img, width, height, &xtrans, &ytrans, &dwidth, &dheight);

	float *output = (float *) malloc(sizeof(float) * dwidth * dheight * 3);
	for (int i = 0; i < dwidth*dheight*3; i++)
	{
		output[i] = 0.5;
	}

	char name [30];
	for (int i = 0; i < num_img; i++)
	{
		apply_transform(input, output, &transforms[9*i], width, height, xtrans, ytrans, dwidth, dheight);
		sprintf(name, "transformed_img_%i.exr", i);
		if (i < 10) sprintf(name, "transformed_img_0%i.exr", i);
		writeOpenEXRFile(name, output, dwidth, dheight);
	}

	free(transforms);
	cudaFree(input);
	free(output);

	return 1;
}
