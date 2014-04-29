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
	runtest(im1, im2, &transforms[9]);

	double im2_2[27] = {0,0,0,-1,-1,1,-2,-2,2,-3,-3,3,-4,-4,4,-5,-5,5,-6,-6,6,-7,-7,7,-8,-8,8};
	runtest(im1, im2_2, &transforms[18]);

	double im2_3[27] = {50,20,0,51,21,1,52,22,2,53,23,3,54,24,4,55,25,5,56,26,6,57,27,7,58,28,8};
	runtest(im1, im2_3, &transforms[27]);

	int width, height;
	float *input;
	readOpenEXRFile(argv[1], &input, width, height);

	int xtrans;
	int ytrans;
	int dwidth;
	int dheight;

	find_dest_multi(transforms, num_img, width, height, &xtrans, &ytrans, &dwidth, &dheight);

	printf("width: %i\nheight: %i\n", width, height);
	printf("xtrans: %i\nytrans: %i\n", xtrans, ytrans);
	printf("dwidth: %i\ndheight: %i\n", dwidth, dheight);

	float *output = (float *) malloc(sizeof(float) * dwidth * dheight * 3);
	for (int i = 0; i < dwidth*dheight*3; i+=3)
	{
		output[i] = 1;
	}

	int arg_index = 2;
	for (int i = 0; i < num_img; i++)
	{
		apply_transform(input, output, &transforms[9*i], width, height, xtrans, ytrans, dwidth, dheight);
		writeOpenEXRFile(argv[arg_index], output, dwidth, dheight);
		arg_index++;
	}

	free(transforms);
	cudaFree(input);
	free(output);

	return 1;
}
