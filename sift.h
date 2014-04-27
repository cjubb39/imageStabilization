//Contains all the kernels necessary for the SIFT algorithm

__global__ void d_grayscale(float *image, float *outArray, int w, int array_size);

void comp_blurs(float *blur_values, int radius);

__global__ void d_gaussian(float *channel, float *blurs, float *outArray, int i, int radius, int w, int h);

__global__ void comp_dog(float *gaussians, float *diffs, int w, int blurs, int array_size);
