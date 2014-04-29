#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

#include <stdio.h>

#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
void gpuCheckError( cudaError_t err,
                         const char *file,
                         int line );

//#define DEBUG

#endif
