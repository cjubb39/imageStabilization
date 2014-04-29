#ifndef MAIN_SIFT_H
#define MAIN_SIFT_H

void sift_images(const char*, const char*, double**, int*, double**, int*);
int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
double ComputeSingular(CudaImage *img, CudaImage *svd);
void GenerateMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img, double **, int *, double **, int *);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

#endif
