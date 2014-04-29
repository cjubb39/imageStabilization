#include "error_handling.h"
//#include "im1.h"

#define rowmajIndex(col, row, width, height) ( ((int) row + height/2)*width + ((int) col + width/2))


/*	
 *	transform_info: cos(angle), sin(angle), trans_x, trans_y
 *	
 */
__global__ void image_transform(float *source, float *destination,
		int width, int height, int xtrans, int ytrans, int dwidth, int dheight,
		float *transform_info){

	/* want origin at center */
	const int x = blockIdx.x * blockDim.x + threadIdx.x - dwidth / 2;
	const int y = blockIdx.y * blockDim.y + threadIdx.y - dheight / 2;
	const int index = rowmajIndex(x, y, dwidth, dheight);

	if (x >= dwidth/2 || y >= dheight/2){
		return;
	}

	/* do translation */
	float fetch_x = x + xtrans + transform_info[2];
	float fetch_y = y + ytrans + transform_info[3];

	/* do rotation */
	float cos_val = transform_info[0], sin_val = transform_info[1];
	
	int tmp = fetch_x;
	fetch_x = tmp*cos_val - fetch_y*sin_val;
	fetch_y = tmp*sin_val + fetch_y*cos_val;

/*printf("Coord: %d, %d: %d; Dim: %d %d" ";; fetch: %d, %d: %d\n", 
	x, y, 3*rowmajIndex(x,y,width, height), width, height, 
	(int) fetch_x, (int) fetch_y, (int) (3*rowmajIndex((int) fetch_x, (int) fetch_y,width, height)));
*/
	if (fetch_x >= width/2 || fetch_x < -width/2 ||
			fetch_y >= height/2 || fetch_y < -height/2){
		destination[3*index] = 0.25;
		destination[3*index + 1] = 0.25;
		destination[3*index + 2] = 0.25;
	} else {
		destination[3*index] =
			source[(int) (3 * rowmajIndex(fetch_x, fetch_y, width, height))];
		destination[3*index + 1] =
			source[(int) (3 * rowmajIndex(fetch_x, fetch_y, width, height) + 1)];
		destination[3*index + 2] =
			source[(int) (3 * rowmajIndex(fetch_x, fetch_y, width, height) + 2)];
	}
}

__host__ void getMaxThreadsPerBlock(int *info){
    int dev_count, highTPB = 0, highdev = 0;
    GPU_CHECKERROR(cudaGetDeviceCount(&dev_count));
    for (int dev = 0; dev < dev_count; dev++){
        cudaDeviceProp deviceProp;
        GPU_CHECKERROR(cudaGetDeviceProperties(&deviceProp, dev));

        if (deviceProp.maxThreadsPerBlock > highTPB){
            highTPB = deviceProp.maxThreadsPerBlock;
            highdev = dev;
        }
    }

    info[0] = highdev;
    info[1] = highTPB;
}


/*	
 *	transform should be in form: (single pointer to array in row major form)
 *		cos(theta)	-sin(theta)	t_x
 *		sin(theta)	cos(theta)	t_y
 *		0			0			1
 */
__host__ void apply_transform(float *input, float *output, double *transform,
		const int width, const int height, int xtrans, int ytrans,
		int dwidth, int dheight){

	 // start the timers
  cudaEvent_t     start, stop;
  float elapsedTime;
  GPU_CHECKERROR( cudaEventCreate( &start ) );
  GPU_CHECKERROR( cudaEventCreate( &stop ) );
  GPU_CHECKERROR( cudaEventRecord( start, 0 ));


	/* convert transform to whats used below */
	float tmp_transform[4];
	tmp_transform[0] = transform[0];
	tmp_transform[1] = -transform[1];
	tmp_transform[2] = transform[2];
	tmp_transform[3] = transform[5];

#ifdef DEBUG
printf("transform prop: %f %f %f %f\n", tmp_transform[0], tmp_transform[1], tmp_transform[2], tmp_transform[3]);
#endif

	float *d_source, *d_destination, *d_transform_info;
	GPU_CHECKERROR(cudaMalloc(&d_source, sizeof(float) * height * width * 3));
	GPU_CHECKERROR(cudaMalloc(&d_destination, sizeof(float) * dheight * dwidth * 3));
	GPU_CHECKERROR(cudaMalloc(&d_transform_info, sizeof(float) * 4));

	GPU_CHECKERROR(cudaMemcpy(d_source, input, 
		sizeof(float) * height * width * 3, cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaMemcpy(d_destination, output, sizeof(float) * dheight * dwidth * 3,
			cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaMemcpy(d_transform_info, tmp_transform,
		sizeof(float) * 4, cudaMemcpyHostToDevice));

	/* run kernel */
	int device_info[2];
	getMaxThreadsPerBlock(device_info);
	GPU_CHECKERROR(cudaSetDevice(device_info[0]));

	int threads_per_block = device_info[1];

	dim3 block_size, grid_size;
	block_size.x = 32; // warp size
	block_size.y = (unsigned int) (threads_per_block / block_size.x);

	grid_size.x = (unsigned int) ((dwidth + block_size.x - 1) / block_size.x);
    grid_size.y = (unsigned int) ((dheight + block_size.y - 1) / block_size.y);

    image_transform<<<grid_size, block_size>>>
    		(d_source, d_destination, width, height, xtrans, ytrans, dwidth,
    				dheight, d_transform_info);
    GPU_CHECKERROR(cudaDeviceSynchronize());

    GPU_CHECKERROR(cudaMemcpy(output, d_destination,
    		sizeof(float) * dheight * dwidth * 3, cudaMemcpyDeviceToHost));
    GPU_CHECKERROR(cudaFree(d_source));
    GPU_CHECKERROR(cudaFree(d_destination));
    GPU_CHECKERROR(cudaFree(d_transform_info));

    GPU_CHECKERROR( cudaEventRecord( stop, 0 ));
    GPU_CHECKERROR( cudaEventSynchronize( stop ) );
    GPU_CHECKERROR( cudaEventElapsedTime( &elapsedTime,
                                      start, stop ) );
    printf( "Time taken:  %3.1f ms\n", elapsedTime );
}

/* Given the set of transformation matrices, find the final destination
 * translation wrt to the original, and size
 */
__host__ void find_dest_multi(double *transforms, int num_transforms, int width,
		int height, int *xtrans, int *ytrans, int *dwidth, int *dheight)
{
	//Start with assuming no transformation
	int xmin = 0;
	int ymin = 0;
	int xmax = width;
	int ymax = height;

	for (int i = 0; i < num_transforms; i++)
	{
		int index = i*9;
		//Find the farthest transform that's above and to the left
		if (transforms[index + 2] < *xtrans)
			*xtrans = transforms[index + 2];
		if (transforms[index + 5] < *ytrans)
			*ytrans = transforms[index + 5];

		//Translate the four corners to find the largest dimensions
		int x1 = -width/2 * transforms[index]
		         - height/2 * transforms[index + 1] + transforms[index + 2] + width/2;
		int y1 = -width/2 * transforms[index + 3]
		         - height/2 * transforms[index + 4] + transforms[index + 5] + height/2;
		int x2 = -width/2 * transforms[index]
		         + height/2 * transforms[index + 1] + transforms[index + 2] + width/2;
		int y2 = -width/2 * transforms[index + 3]
		         + height/2 * transforms[index + 4] + transforms[index + 5] + height/2;
		int x3 = width/2 * transforms[index]
		         - height/2 * transforms[index + 1] + transforms[index + 2] + width/2;
		int y3 = width/2 * transforms[index + 3]
		         - height/2 * transforms[index + 4] + transforms[index + 5] + height/2;
		int x4 = width/2 * transforms[index]
		         + height/2 * transforms[index + 1] + transforms[index + 2] + width/2;
		int y4 = width/2 * transforms[index + 3]
		         + height/2 * transforms[index + 4] + transforms[index + 5] + height/2;

		xmin = min(min(min(min(x1, x2), x3), x4), xmin);
		ymin = min(min(min(min(y1, y2), y3), y4), ymin);
		xmax = max(max(max(max(x1, x2), x3), x4), xmax);
		ymax = max(max(max(max(y1, y2), y3), y4), ymax);
	}

	*xtrans = -(*xtrans)/2;
	*ytrans = -(*ytrans)/2;

	*dwidth = xmax - xmin;
	*dheight = ymax - ymin;
}
