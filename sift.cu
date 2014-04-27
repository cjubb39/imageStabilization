#include "sift.h"
#include <math.h>

__global__ void d_grayscale(float *image, float *outArray, int w, int array_size)
{
	//Compute position in 2D array
	int thread_row = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_column = blockDim.x*blockIdx.x + threadIdx.x;

	//Convert to flattened array
	int array_pos = (w * thread_row + thread_column)*3;

	if (array_pos < array_size)
	{
	    float L = 0.2126f*image[array_pos] +
	              0.7152f*image[array_pos+1] +
	              0.0722f*image[array_pos+2];

	    outArray[array_pos/3] = L;
	}

	return;
}

//Compute blur values
void comp_blurs(float *blur_values, int radius)
{
	float scalar = 1.0 / (2 * M_PI * (radius * radius / 9.0));
	float exp_scalar = 1.0 / (2 * (radius * radius / 9.0));
	for (int x = -radius; x <= radius; x++)
	{
		for (int y = -radius; y <= radius; y++)
		{
			int xi = x + radius;
			int yi = y + radius;
			blur_values[xi*(2*radius+1) + yi] = scalar * exp(-(x*x+y*y)*exp_scalar);
		}
	}
}

__global__ void d_gaussian(float *channel, float *blurs, float *outArray, int i, int radius, int w, int h)
{
	//Compute position in 2D array
	int thread_row = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_column = blockDim.x*blockIdx.x + threadIdx.x;
	int tile_width = 2*(radius) + 1;

	int array_pos = w * thread_row + thread_column;

	if (array_pos < w * h)
	{
		for (int x = -radius; x <= radius; x++)
		{
			for (int y = -radius; y <= radius; y++)
			{
				int clipped_row = min(max(thread_row + x, 0), h-1);
				int clipped_column = min(max(thread_column + y, 0), w-1);
				int index = w*clipped_row + clipped_column;
				float pixel = channel[index];
				float blur_weight = blurs[(x + radius) * tile_width + y + radius];

				outArray[array_pos + i*w*h] += pixel * blur_weight;
			}
		}
	}

	return;
}

__global__ void comp_dog(float *gaussians, float *diffs, int w, int k_1, int array_size)
{
	//Compute position in 2D array
	int thread_row = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_column = blockDim.x*blockIdx.x + threadIdx.x;

	//Convert to flattened array
	int array_pos = w * thread_row + thread_column;

	if (array_pos < array_size)
	{
		for (int i = 0; i < k_1; i++)
		{
			int new_pos = array_pos + i * array_size;
			int next_pos = array_pos + (i+1) * array_size;
			diffs[new_pos] = gaussians[new_pos] - gaussians[next_pos];
		}
	}

	return;
}

__global__ void comp_extrema(float *diff_gauss, int *extremas, int w, int h, int layers, int array_size)
{
	//Compute position in 2D array
	int thread_row = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_column = blockDim.x*blockIdx.x + threadIdx.x;
	int layer = blockDim.z*blockIdx.z + threadIdx.z;

	//Convert to flattened array
	int array_pos = w * thread_row + thread_column + array_size*layer;

	if (array_pos < array_size*layers)
	{
		float minimum = 255;
		float maximum = 0;
		//Go through the three layers
		for (int l = max(array_pos, array_pos - array_size); l < min(array_pos, array_pos + array_size*2); l += array_size)
		{
			//3x3 square
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					int clipped_row = min(max(thread_row + i, 0), h-1);
					int clipped_column = min(max(thread_column + j, 0), w-1);

					int index = w*clipped_row + clipped_column;
					minimum = min(minimum, diff_gauss[index]);
					maximum = max(maximum, diff_gauss[index]);
				}
			}
		}

		//See if the current pixel is an extrema
		extremas[array_pos] = ((minimum == diff_gauss[array_pos]) || (maximum == diff_gauss[array_pos])) ? 1 : 0;
	}

	return;
}
