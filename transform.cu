#include "error_handling.h"
//#include "im1.h"

#define rowmajIndex(col, row, width, height) ( ((int) row)*width + ((int) col))

__host__ void comb_trans(double *transforms, int num_trans)
{
	//Multiply consecutive transforms together so all will eventually be transformed to the original matrix
	for(int k = 0; k < num_trans-1; k++)
	{
		double a, b, c, d, e, f, g, h, i;
		a = transforms[9*(k+1)] 	* transforms[9*k] +
			transforms[9*(k+1) + 1] * transforms[9*k + 3] +
			transforms[9*(k+1) + 2] * transforms[9*k + 6];
		b = transforms[9*(k+1)] 	* transforms[9*k + 1] +
			transforms[9*(k+1) + 1] * transforms[9*k + 4] +
			transforms[9*(k+1) + 2] * transforms[9*k + 7];
		c = transforms[9*(k+1)]		* transforms[9*k + 2] +
			transforms[9*(k+1) + 1] * transforms[9*k + 5] +
			transforms[9*(k+1) + 2] * transforms[9*k + 8];
		d = transforms[9*(k+1) + 3] * transforms[9*k] +
			transforms[9*(k+1) + 4] * transforms[9*k + 3] +
			transforms[9*(k+1) + 5] * transforms[9*k + 6];
		e = transforms[9*(k+1) + 3] * transforms[9*k + 1] +
			transforms[9*(k+1) + 4] * transforms[9*k + 4] +
			transforms[9*(k+1) + 5] * transforms[9*k + 7];
		f = transforms[9*(k+1) + 3] * transforms[9*k + 2] +
			transforms[9*(k+1) + 4] * transforms[9*k + 5] +
			transforms[9*(k+1) + 5] * transforms[9*k + 8];
		g = transforms[9*(k+1) + 6] * transforms[9*k] +
			transforms[9*(k+1) + 7] * transforms[9*k + 3] +
			transforms[9*(k+1) + 8] * transforms[9*k + 6];
		h = transforms[9*(k+1) + 6] * transforms[9*k + 1] +
			transforms[9*(k+1) + 7] * transforms[9*k + 4] +
			transforms[9*(k+1) + 8] * transforms[9*k + 7];
		i = transforms[9*(k+1) + 6] * transforms[9*k + 2] +
			transforms[9*(k+1) + 7] * transforms[9*k + 5] +
			transforms[9*(k+1) + 8] * transforms[9*k + 8];
		transforms[9*(k+1)] 	= a;
		transforms[9*(k+1) + 1] = b;
		transforms[9*(k+1) + 2] = c;
		transforms[9*(k+1) + 3] = d;
		transforms[9*(k+1) + 4] = e;
		transforms[9*(k+1) + 5] = f;
		transforms[9*(k+1) + 6] = g;
		transforms[9*(k+1) + 7] = h;
		transforms[9*(k+1) + 8] = i;
	}
}

/*	
 *	transform_info: cos(angle), sin(angle), trans_x, trans_y
 *	
 */
__global__ void image_transform(float *source, float *destination,
		int width, int height, int xtrans, int ytrans, int dwidth, int dheight,
		float *transform_info){

	/* want origin at center */
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = rowmajIndex(x, y, dwidth, dheight);

/*	if (x >= dwidth || y >= dheight || x < 0 || y < 0){
		return;
	}*/

	if ((x) >= (dwidth) || (y) >= (dheight) || (x) < 0 || (y) < 0){
		return;
	}

	/* do translation */
	float fetch_x = x - xtrans;
	float fetch_y = y - ytrans;

	/* do rotation */
/*	float cos_val = transform_info[0], sin_val = transform_info[1];
	
	fetch_x = tmp*cos_val - fetch_y*sin_val;// - transform_info[2];
	fetch_y = tmp*sin_val + fetch_y*cos_val;// - transform_info[3];*/

	/* apply transform */
	float tmp = fetch_x;
	float divisor = fetch_x*transform_info[6] + fetch_y*transform_info[7] + transform_info[8];
	fetch_x = (tmp * transform_info[0] + fetch_y * transform_info[1] + transform_info[2])
		/ divisor;
	fetch_y = (tmp * transform_info[3] + fetch_y * transform_info[4] + transform_info[5])
		/ divisor;

	//fetch_x += xtrans;
	//fetch_y += ytrans;

/*printf("Coord: %d, %d: %d; Dim: %d %d" ";; fetch: %d, %d: %d\n", 
	x, y, 3*rowmajIndex(x,y,width, height), width, height, 
	(int) fetch_x, (int) fetch_y, (int) (3*rowmajIndex((int) fetch_x, (int) fetch_y,width, height)));
*/
/*	if (fetch_x >= width/2 || fetch_x < -width/2 ||
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
	}*/

	if (fetch_x >= width || fetch_x < 0 ||
			fetch_y >= height || fetch_y < 0){
		destination[3*index] = 0;
		destination[3*index + 1] = 0;
		destination[3*index + 2] = 0;
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


	/* convert transform to whats used below 
		 cos(theta), sin(theta), dx, dy */
/*	float tmp_transform[4];
	tmp_transform[0] = transform[0];
	tmp_transform[1] = -transform[1];
	tmp_transform[2] = transform[2];
	tmp_transform[3] = transform[5];*/
	float f_transform[9];
	for(int i = 0; i < 9; ++i){
		f_transform[i] = transform[i];
	}

	float *d_source, *d_destination, *d_transform_info;
	GPU_CHECKERROR(cudaMalloc(&d_source, sizeof(float) * height * width * 3));
	GPU_CHECKERROR(cudaMalloc(&d_destination, sizeof(float) * dheight * dwidth * 3));
	GPU_CHECKERROR(cudaMalloc(&d_transform_info, sizeof(float) * 9));

	GPU_CHECKERROR(cudaMemcpy(d_source, input, 
		sizeof(float) * height * width * 3, cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaMemcpy(d_destination, output, sizeof(float) * dheight * dwidth * 3,
			cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaMemcpy(d_transform_info, f_transform,
		sizeof(float) * 9, cudaMemcpyHostToDevice));

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
    printf( "Apply Transform Time taken:  %3.2f ms\n", elapsedTime );
}

/* Given the set of transformation matrices, find the final destination
 * translation wrt to the original, and size
 */
__host__ void find_dest_multi(double *transforms, int num_transforms, int width,
		int height, int *xtrans, int *ytrans, int *dwidth, int *dheight)
{
	//Start with assuming no transformation
	double xmin = width;
	double ymin = height;
	double xmax = 0;
	double ymax = 0;

	*xtrans = 0;
	*ytrans = 0;

	int xtrans_0;
	int ytrans_0;

	for (int i = 0; i < num_transforms; i++)
	{
		int index = i*9;


		double inverted_matrix[9];
		double det = transforms[index] * transforms[index + 4] * transforms[index + 8] +
			transforms[index + 3] * transforms[index + 7] * transforms[index + 2] +
			transforms[index + 6] * transforms[index + 1] * transforms[index + 5] -
			transforms[index + 1] * transforms[index + 7] * transforms[index + 5] -
			transforms[index + 6] * transforms[index + 4] * transforms[index + 2] -
			transforms[index + 3] * transforms[index + 1] * transforms[index + 8];

		inverted_matrix[0] = transforms[index + 4] * transforms[index + 8] -
			transforms[index + 5]*transforms[index + 7];

		inverted_matrix[1] = transforms[index + 2] * transforms[index + 7] -
			transforms[index + 1]*transforms[index + 8];

		inverted_matrix[2] = transforms[index + 1] * transforms[index + 5] -
			transforms[index + 2]*transforms[index + 4];

		inverted_matrix[3] = transforms[index + 5] * transforms[index + 6] -
			transforms[index + 3]*transforms[index + 8];

		inverted_matrix[4] = transforms[index] * transforms[index + 8] -
			transforms[index + 2]*transforms[index + 6];

		inverted_matrix[5] = transforms[index + 2] * transforms[index + 3] -
			transforms[index]*transforms[index + 5];

		inverted_matrix[6] = transforms[index + 3] * transforms[index + 7] -
			transforms[index + 4]*transforms[index + 6];

		inverted_matrix[7] = transforms[index + 1] * transforms[index + 6] -
			transforms[index]*transforms[index + 7];

		inverted_matrix[8] = transforms[index] * transforms[index + 4] -
			transforms[index + 1]*transforms[index + 3];

		for (int j = 0; j < 9; ++j)
			inverted_matrix[j] /= det;


/*		//Find the farthest transform that's above and to the left
		if (-transforms[index + 2] < *xtrans)
			*xtrans = -transforms[index + 2];
		if (-transforms[index + 5] < *ytrans)
			*ytrans = -transforms[index + 5];*/


		/*
		 *	x1-------x4
		 *	|         |
		 *	|         |
		 *	x2-------x3
		 */

		//Translate the four corners to find the largest dimensions
		double pt1_scale = inverted_matrix[8];
		double x1 = inverted_matrix[2] / pt1_scale;
		double y1 = inverted_matrix[5] / pt1_scale;

		double pt2_scale = height * inverted_matrix[7] + inverted_matrix[8];
		double x2 = (height * inverted_matrix[1] + inverted_matrix[2]) / pt2_scale;
		double y2 = (height * inverted_matrix[4] + inverted_matrix[5]) / pt2_scale;
		
		double pt3_scale = width * inverted_matrix[6] + pt2_scale;
		double x3 = (width * transforms[index] + height * inverted_matrix[1]
			+ inverted_matrix[2]) / pt3_scale;
		double y3 = (width * inverted_matrix[3] + height * inverted_matrix[4]
			+ inverted_matrix[5]) / pt3_scale;
		
		double pt4_scale = width * inverted_matrix[6] + inverted_matrix[8];
		double x4 = (width * transforms[index] + inverted_matrix[2]) / pt4_scale;
		double y4 = (width * inverted_matrix[3] + inverted_matrix[5]) / pt4_scale;

		xmin = min(min(min(min(x1, x2), x3), x4), xmin);
		ymin = min(min(min(min(y1, y2), y3), y4), ymin);
		xmax = max(max(max(max(x1, x2), x3), x4), xmax);
		ymax = max(max(max(max(y1, y2), y3), y4), ymax);

		if (i == 0){
			xtrans_0 = xmin;
			ytrans_0 = ymin;
		}
	}

	*dwidth = (int) xmax - (int) xmin + 20;
	*dheight = (int) ymax - (int) ymin + 20;

	*xtrans = (int) xtrans_0 - (int) xmin + 10;
	*ytrans = (int) ytrans_0 - (int) ymin + 10;
}
