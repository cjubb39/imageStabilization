#ifndef TRANS
#define TRANS

//Transform functions
__host__ void comb_trans(double *transforms, int num_trans);
__global__ void image_transform(float *source, float *destination,
		int width, int height, int xtrans, int ytrans, int dwidth, int dheight,
		float *transform_info);
__host__ void getMaxThreadsPerBlock(int *info);
__host__ void apply_transform(float *input, float *output, double *transform,
		const int width, const int height, int xtrans, int ytrans,
		int dwidth, int dheight);
__host__ void find_dest_multi(double *transforms, int num_transforms, int width,
		int height, int *xtrans, int *ytrans, int *dwidth, int *dheight);


#endif
