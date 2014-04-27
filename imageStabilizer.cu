//Chae Jubb and Olivia Winn

#define HAS_OPENEXR 1

#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

#include "im1.h"
#include "sift.h"

// handy error macro:
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                          const char *file,
                          int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
               file, line );
        exit( EXIT_FAILURE );
    }
}

int main (int argc, char *argv[])
{
	//For now, read in a single image
    printf("reading openEXR file %s\n", argv[1]);       
    int w, h;   
    float *h_imageArray;
    readOpenEXRFile (argv[1], &h_imageArray, w, h);
    
    int rgb_arraySize = 3*w*h;
    int arraySize = w*h;
    
    //Copy to GPU
    float *d_imageArray;
    GPU_CHECKERROR(
    	cudaMalloc((void **) &d_imageArray, rgb_arraySize * sizeof(float))
    );
    GPU_CHECKERROR(
    	cudaMemcpy((void *) d_imageArray, h_imageArray, rgb_arraySize * sizeof(float), cudaMemcpyHostToDevice)
    );
    
	//Set number of threads and blocks
	//Get device information and pick one with max threads per block
	//(code from textbook)
	int dev_count;
	unsigned int threads_per_block = 0;
	int best_device = -1;
	GPU_CHECKERROR(
		cudaGetDeviceCount(&dev_count)
	);
	cudaDeviceProp dev_prop;
	for (int i = 0; i < dev_count; i++)
	{
		GPU_CHECKERROR(
			cudaGetDeviceProperties(&dev_prop, i)
		);
		if (dev_prop.maxThreadsPerBlock > threads_per_block)
		{
			threads_per_block = dev_prop.maxThreadsPerBlock;
			best_device = i;
		}
	}
	cudaSetDevice(best_device);

	int x_threads = 32;
	int y_threads = floor(threads_per_block / x_threads);
	int x_blocks = ceil (w / (float) x_threads);
	int y_blocks = ceil (h / (float) y_threads);

    //Convert to grayscale
	float *d_grayArray;
	GPU_CHECKERROR(
		cudaMalloc((void **) &d_grayArray, arraySize * sizeof(float))
	);
	GPU_CHECKERROR(
		cudaMemset((void *) d_grayArray, 0, arraySize * sizeof(float))
	);

    d_grayscale<<<dim3(x_blocks, y_blocks), dim3(x_threads, y_threads)>>>(d_imageArray, d_grayArray, w, rgb_arraySize);

    //Number of gaussian blurs to apply
    int k = 5;

    //Make an array to hold all the blurred images
    float *d_gaussArray;
    GPU_CHECKERROR(
    	cudaMalloc((void **) &d_gaussArray, arraySize * k * sizeof(float))
    );
    GPU_CHECKERROR(
    	cudaMemset((void *) d_gaussArray, 0, arraySize * k * sizeof(float))
    );

    int sigma = 6;
    int radius = sigma / 2;
    int tile_width = 2*radius + 1;

    float *blurs;
    float *d_blurs;

    //Create the 'stack' of Gaussian blur images
    for (int i = 0; i < k; i++)
    {
    	//Create the blur matrix
    	GPU_CHECKERROR(
    		cudaMallocHost((void **) &blurs, tile_width * tile_width * sizeof(float))
    	);
    	GPU_CHECKERROR(
    		cudaMalloc((void **) &d_blurs, tile_width * tile_width * sizeof(float))
    	);

    	comp_blurs(blurs, radius);


    	GPU_CHECKERROR(
    		cudaMemcpy((void *) d_blurs, blurs, tile_width * tile_width * sizeof(float), cudaMemcpyHostToDevice)
    	);

    	//Compute the new image
    	d_gaussian<<<dim3(x_blocks, y_blocks), dim3(x_threads, y_threads)>>>
    			(d_grayArray, d_blurs, d_gaussArray, i, radius, w, h);

    	//Increase sigma for the next blur
    	radius = sigma;
    	sigma *= 2;
    	tile_width = sigma + 1;

    	//Free the blurs so they can be reset for the next iteration
    	cudaFreeHost(blurs);
    	cudaFree(d_blurs);
    }

    //Convert to DoG
    float *diff_gauss;
    GPU_CHECKERROR(
    	cudaMalloc((void **) &diff_gauss, arraySize * (k-1) * sizeof(float))
    );
    GPU_CHECKERROR(
    	cudaMemset((void *) diff_gauss, 0, arraySize * (k-1) * sizeof(float))
    );

    comp_dog<<<dim3(x_blocks, y_blocks), dim3(x_threads, y_threads)>>>
    		(d_gaussArray, diff_gauss, w, k-1, arraySize);

//    //Save one image to test
//    float *h_testArray = (float *) malloc(arraySize * sizeof(float));
//    GPU_CHECKERROR(
//    	cudaMemcpy((void *) h_testArray, diff_gauss, arraySize * sizeof(float), cudaMemcpyDeviceToHost)
//    );
//
//    for (int i = 0; i < arraySize; i++)
//    {
//    	h_imageArray[3*i] = h_testArray[i];
//    	h_imageArray[3*i+1] = h_testArray[i];
//    	h_imageArray[3*i+2] = h_testArray[i];
//    }
//
//    writeOpenEXRFile ("gauss_test.exr", h_imageArray, w, h);

    //End the program
    cudaFreeHost(h_imageArray);
    cudaFree(d_imageArray);
    cudaFree(d_gaussArray);
    cudaFree(diff_gauss);
    cudaFreeHost(h_testArray);
    
    printf("done.\n");

    return 0;
}
