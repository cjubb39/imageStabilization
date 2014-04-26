#include "sift.h"

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

	    outArray[array_pos] = L;
	    outArray[array_pos+1] = L;
	    outArray[array_pos+2] = L;
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

__global__ void d_gaussian(float *channel, float *blurs, float *outArray, int radius, int w, int h)
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

				outArray[array_pos] += pixel * blur_weight;
			}
		}
	}

	return;
}
